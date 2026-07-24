using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributions
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using Printf
using ProgressMeter
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

include(joinpath(@__DIR__, "initial_state.jl"))

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "regret_transfer_benchmark_final_iter")
const DEFAULT_FITTED_MODELS = joinpath(
    @__DIR__,
    "regret_fit_results_final_iter",
    "models.jld2",
)

softplus_output(x) = Flux.softplus.(x)

function with_softplus_output(actor)
    actor isa Chain || error(
        "Softplus refit expects each checkpoint regret actor to be a Flux.Chain",
    )
    return Chain(actor.layers..., softplus_output)
end

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function checkpoint_iteration(path)
    matched = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(matched) && error("Invalid checkpoint name: $(path)")
    return parse(Int, only(matched.captures))
end

function select_checkpoint(spec::AbstractString)
    models_dir = joinpath(@__DIR__, "models_$(SEARCH_NAME)")
    checkpoints = filter(
        path -> occursin(r"oracle\d+\.jld2$", basename(path)),
        readdir(models_dir; join=true),
    )
    isempty(checkpoints) && error("No checkpoints found in $(models_dir)")
    sort!(checkpoints; by=checkpoint_iteration)
    spec == "latest" && return last(checkpoints)
    requested = parse(Int, spec)
    return only(filter(path -> checkpoint_iteration(path) == requested, checkpoints))
end

function build_search(oracle; queries, max_depth, search_epsilon, prior_scale)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
        value_target=:search,
        ϵ=_ -> search_epsilon,
        prior_scale,
    )
end

function evaluate_arm(game, initialstates, oracle, solver_name, search_player;
                      runs, max_steps, queries, max_depth, search_epsilon,
                      prior_scale, seed)
    search = build_search(
        oracle;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    planner = AZ.AlphaZeroPlanner(game, search)
    search_policy = Tools.SinglePlayerAlphaZeroPolicy(planner, search_player)
    heuristic_policy = Tools.sda_no_burn_heuristic(game, 3 - search_player)
    joint_policy = search_player == 1 ?
        Tools.JointPolicy(search_policy, heuristic_policy) :
        Tools.JointPolicy(heuristic_policy, search_policy)

    @printf(
        "[transfer-benchmark] starting solver=%s player=%d runs=%d prior_scale=%.1f\n",
        solver_name,
        search_player,
        runs,
        prior_scale,
    )
    flush(stdout)
    Random.seed!(seed + search_player)
    elapsed = @elapsed result = Tools.evaluate_joint_policy(
        game,
        joint_policy,
        runs;
        max_steps,
        initialstates,
        show_progress=true,
        proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), SDAOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(
                :reward;
                name=:stderr_reward,
                init=zero(MarkovGames.reward_type(game)),
            ),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
    )
    @printf(
        "[transfer-benchmark] finished solver=%s player=%d reward=%.6f stderr=%.6f elapsed=%.1fs\n",
        solver_name,
        search_player,
        result.reward[search_player],
        result.stderr_reward[search_player],
        elapsed,
    )
    flush(stdout)
    return (;
        solver=solver_name,
        search_player,
        runs,
        reward=result.reward[search_player],
        stderr_reward=result.stderr_reward[search_player],
        mean_steps=result.mean_steps,
        detection_rate=result.detected_rate,
        target_escaped_rate=result.target_escaped_rate,
        observer_lost_rate=result.observer_lost_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        prior_scale,
        elapsed_seconds=elapsed,
    )
end

function csv_value(value)
    value isa AbstractString && return value
    value isa Real && return isfinite(value) ? string(value) : ""
    return string(value)
end

function write_csv(path, rows)
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
    return path
end

function main(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 200, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 100, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    prior_scale = option_value(args, "--prior-scale", 100.0, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260722, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    output_dir = abspath(option_value(args, "--output-dir", DEFAULT_OUTPUT_DIR, String))
    fitted_models_path = abspath(option_value(
        args,
        "--fitted-models",
        DEFAULT_FITTED_MODELS,
        String,
    ))
    refit_activation = Symbol(lowercase(option_value(
        args,
        "--refit-activation",
        "linear",
        String,
    )))
    if test
        runs = min(runs, 4)
        queries = min(queries, 4)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 5)
        prior_scale = min(prior_scale, Float64(queries))
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    max_steps > 0 || error("--max-steps must be positive")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")
    0 <= prior_scale <= queries || error("--prior-scale must be in [0, tree_queries]")
    refit_activation in (:linear, :softplus) || error(
        "--refit-activation must be linear or softplus",
    )
    isfile(fitted_models_path) || error("Missing fitted models $(fitted_models_path)")

    checkpoint = select_checkpoint(checkpoint_spec)
    oracle_file = joinpath(@__DIR__, "oracle_$(SEARCH_NAME).jld2")
    online_oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(online_oracle, checkpoint)
    refit_oracle = if refit_activation == :linear
        deepcopy(online_oracle)
    else
        regret_head = AZ.MultiActor(
            with_softplus_output(deepcopy(online_oracle.regret_head[1])),
            with_softplus_output(deepcopy(online_oracle.regret_head[2])),
        )
        AZ.FittedRegretModel(
            deepcopy(online_oracle.shared),
            regret_head,
            deepcopy(online_oracle.strategy_head),
            deepcopy(online_oracle.critic);
            value_weight=online_oracle.value_weight,
            regret_weight=online_oracle.regret_weight,
            strategy_weight=online_oracle.strategy_weight,
        )
    end
    fitted = JLD2.load(fitted_models_path)
    Flux.loadmodel!(refit_oracle.regret_head[1], fitted["baseline_p1_state"])
    Flux.loadmodel!(refit_oracle.regret_head[2], fitted["baseline_p2_state"])

    game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
    initial_distribution = core_initialstate_distribution(game)
    initial_rng = MersenneTwister(seed)
    initialstates = [rand(initial_rng, initial_distribution) for _ in 1:runs]

    println("[transfer-benchmark] checkpoint=$(checkpoint)")
    println(
        "[transfer-benchmark] refit=$(fitted_models_path):baseline_p1/p2 ",
        "activation=$(refit_activation)",
    )
    println(
        "[transfer-benchmark] runs=$(runs) queries=$(queries) depth=$(max_depth) ",
        "steps=$(max_steps) search_epsilon=$(search_epsilon) action_epsilon=0.0",
    )
    flush(stdout)

    rows = NamedTuple[]
    refit_solver_name = refit_activation == :linear ?
        "refit_regret_transfer" :
        "softplus_refit_regret_transfer"
    arms = (
        ("value_only", online_oracle, 0.0),
        ("online_transfer", online_oracle, prior_scale),
        (refit_solver_name, refit_oracle, prior_scale),
    )
    for search_player in 1:2
        for (solver_name, oracle, arm_prior_scale) in arms
            push!(rows, evaluate_arm(
                game,
                initialstates,
                oracle,
                solver_name,
                search_player;
                runs,
                max_steps,
                queries,
                max_depth,
                search_epsilon,
                prior_scale=arm_prior_scale,
                seed,
            ))
        end
    end

    value_only_rewards = Dict(
        row.search_player => row.reward for row in rows if row.solver == "value_only"
    )
    summary_rows = map(rows) do row
        merge(row, (delta_vs_value_only=row.reward - value_only_rewards[row.search_player],))
    end
    mkpath(output_dir)
    output = write_csv(joinpath(output_dir, "heuristic_matchups.csv"), summary_rows)
    println("[transfer-benchmark] wrote $(output)")
    return summary_rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
