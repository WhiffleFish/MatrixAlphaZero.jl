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
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

include(joinpath(@__DIR__, "initial_state.jl"))
include(joinpath(@__DIR__, "fit_regret_hurdle.jl"))

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const DEFAULT_FITTED_MODELS = joinpath(
    @__DIR__,
    "regret_fit_results_softplus_long",
    "models.jld2",
)
const DEFAULT_OUTPUT_DIR = joinpath(
    @__DIR__,
    "regret_transfer_scheme_screen",
)

softplus_refit_output(x) = Flux.softplus.(x)

function with_softplus_output(actor)
    actor isa Chain || error("Expected checkpoint regret actor to be a Flux.Chain")
    return Chain(actor.layers..., softplus_refit_output)
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

struct TransformedRegretOracle{O,R}
    base::O
    regressors::R
    source::Symbol
    transform::Symbol
    threshold::Float64
    exponent::Float64
    clip::Float64
end

AZ.value(oracle::TransformedRegretOracle, input) = AZ.value(oracle.base, input)
AZ.state_value(oracle::TransformedRegretOracle, game, state) =
    AZ.state_value(oracle.base, game, state)
AZ.batch_state_value(oracle::TransformedRegretOracle, game, states) =
    AZ.batch_state_value(oracle.base, game, states)
AZ.state_strategy(oracle::TransformedRegretOracle, game, state) =
    AZ.state_strategy(oracle.base, game, state)
AZ.batch_state_strategy(oracle::TransformedRegretOracle, game, states) =
    AZ.batch_state_strategy(oracle.base, game, states)

function raw_regret(oracle::TransformedRegretOracle, game, state)
    if oracle.source == :baseline
        return AZ.state_regret(oracle.base, game, state)
    elseif oracle.source == :hurdle
        input = MarkovGames.convert_s(Vector{Float32}, state, game)
        return map(oracle.regressors) do regressor
            vec(hurdle_outputs(regressor.model, input, regressor.tau).prediction)
        end
    end
    error("Unsupported regret source $(oracle.source)")
end

function transform_regret(raw, transform, threshold, exponent, clip)
    regret = clamp.(Float64.(raw), 0.0, clip)
    if transform == :raw
        return regret
    elseif transform == :hard_threshold
        regret[regret .< threshold] .= 0.0
        return regret
    elseif transform == :soft_threshold
        return max.(regret .- threshold, 0.0)
    elseif transform == :top1
        output = zeros(Float64, length(regret))
        isempty(regret) || (output[argmax(regret)] = maximum(regret))
        return output
    elseif transform == :l1
        total = sum(regret)
        return total > 0 ? regret ./ total : regret
    elseif transform == :power_l1
        regret .= regret .^ exponent
        total = sum(regret)
        return total > 0 ? regret ./ total : regret
    elseif transform == :gap_l1
        total = sum(regret)
        total > 0 || return regret
        ordered = sort(regret; rev=true)
        gap = length(ordered) > 1 ?
            (ordered[1] - ordered[2]) / max(ordered[1], eps(Float64)) :
            1.0
        return gap .* regret ./ total
    end
    error("Unsupported regret transform $(transform)")
end

function AZ.state_regret(oracle::TransformedRegretOracle, game, state)
    raw = raw_regret(oracle, game, state)
    return map(raw) do regret
        transform_regret(
            regret,
            oracle.transform,
            oracle.threshold,
            oracle.exponent,
            oracle.clip,
        )
    end
end

function build_refit_oracles(online_oracle, fitted_models_path)
    fitted = JLD2.load(fitted_models_path)
    baseline_oracle = AZ.FittedRegretModel(
        deepcopy(online_oracle.shared),
        AZ.MultiActor(
            with_softplus_output(deepcopy(online_oracle.regret_head[1])),
            with_softplus_output(deepcopy(online_oracle.regret_head[2])),
        ),
        deepcopy(online_oracle.strategy_head),
        deepcopy(online_oracle.critic);
        value_weight=online_oracle.value_weight,
        regret_weight=online_oracle.regret_weight,
        strategy_weight=online_oracle.strategy_weight,
    )
    Flux.loadmodel!(baseline_oracle.regret_head[1], fitted["baseline_p1_state"])
    Flux.loadmodel!(baseline_oracle.regret_head[2], fitted["baseline_p2_state"])

    metadata = fitted["metadata"]
    dataset = JLD2.load(metadata["dataset"])
    input_dim = size(dataset["states"], 1)
    width = Int(metadata["width"])
    tau = Float64(metadata["tau"])
    output_dims = size(dataset["regret_p1"], 1), size(dataset["regret_p2"], 1)
    hurdle_models = map(output_dims) do output_dim
        HurdleRegressor(input_dim, width, output_dim)
    end
    Flux.loadmodel!(hurdle_models[1], fitted["hurdle_p1_state"])
    Flux.loadmodel!(hurdle_models[2], fitted["hurdle_p2_state"])
    hurdle_regressors = map(hurdle_models) do model
        (; model, tau=Float32(tau))
    end
    return baseline_oracle, hurdle_regressors, metadata
end

function candidate(
        name;
        source=:baseline,
        transform=:raw,
        scale=10.0,
        threshold=0.0,
        exponent=1.0,
        clip=Inf,
        reach_power=1.0,
        regret_weight=1.0,
        strategy_weight=0.0,
        statistic_weight=0.0,
    )
    return (;
        name,
        source,
        transform,
        scale=Float64(scale),
        threshold=Float64(threshold),
        exponent=Float64(exponent),
        clip=Float64(clip),
        reach_power=Float64(reach_power),
        regret_weight=Float64(regret_weight),
        strategy_weight=Float64(strategy_weight),
        statistic_weight=Float64(statistic_weight),
    )
end

function screening_candidates()
    candidates = [
        candidate("value_only"; scale=0.0, regret_weight=0.0),
        candidate(
            "coupled_full_scale10";
            scale=10.0,
            regret_weight=1.0,
            strategy_weight=1.0,
            statistic_weight=1.0,
        ),
    ]
    for scale in (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)
        push!(candidates, candidate("baseline_raw_s$(scale)"; scale))
    end
    for reach_power in (0.0, 0.5, 2.0)
        push!(candidates, candidate(
            "baseline_raw_s10_reach$(reach_power)";
            scale=10.0,
            reach_power,
        ))
    end
    for threshold in (0.025, 0.05, 0.1, 0.2)
        push!(candidates, candidate(
            "baseline_hard$(threshold)_s10";
            scale=10.0,
            transform=:hard_threshold,
            threshold,
        ))
        push!(candidates, candidate(
            "baseline_soft$(threshold)_s10";
            scale=10.0,
            transform=:soft_threshold,
            threshold,
        ))
    end
    for scale in (2.5, 5.0, 10.0, 25.0)
        push!(candidates, candidate(
            "baseline_top1_s$(scale)";
            scale,
            transform=:top1,
        ))
        push!(candidates, candidate(
            "baseline_l1_s$(scale)";
            scale,
            transform=:l1,
        ))
    end
    for exponent in (1.5, 2.0, 4.0)
        push!(candidates, candidate(
            "baseline_power$(exponent)_s10";
            scale=10.0,
            transform=:power_l1,
            exponent,
        ))
    end
    for scale in (10.0, 25.0, 50.0)
        push!(candidates, candidate(
            "baseline_gap_l1_s$(scale)";
            scale,
            transform=:gap_l1,
        ))
    end
    for transform in (:raw, :top1, :l1, :gap_l1)
        for scale in (5.0, 10.0, 25.0)
            push!(candidates, candidate(
                "hurdle_$(transform)_s$(scale)";
                source=:hurdle,
                transform,
                scale,
            ))
        end
    end
    return candidates
end

function build_search(
        oracle,
        config;
        queries,
        max_depth,
        search_epsilon,
    )
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
        value_target=:search,
        ϵ=_ -> search_epsilon,
        prior_scale=config.scale,
        regret_prior_weight=config.regret_weight,
        strategy_prior_weight=config.strategy_weight,
        statistic_prior_weight=config.statistic_weight,
        prior_reach_power=config.reach_power,
    )
end

function evaluate_candidate(
        game,
        initialstates,
        baseline_oracle,
        hurdle_regressors,
        config,
        search_player;
        max_steps,
        queries,
        max_depth,
        search_epsilon,
        seed,
    )
    oracle = TransformedRegretOracle(
        baseline_oracle,
        hurdle_regressors,
        config.source,
        config.transform,
        config.threshold,
        config.exponent,
        config.clip,
    )
    search = build_search(
        oracle,
        config;
        queries,
        max_depth,
        search_epsilon,
    )
    planner = AZ.AlphaZeroPlanner(game, search)
    search_policy = Tools.SinglePlayerAlphaZeroPolicy(planner, search_player)
    heuristic_policy = Tools.sda_no_burn_heuristic(game, 3 - search_player)
    joint_policy = search_player == 1 ?
        Tools.JointPolicy(search_policy, heuristic_policy) :
        Tools.JointPolicy(heuristic_policy, search_policy)

    @printf(
        "[scheme-screen] start candidate=%s player=%d runs=%d\n",
        config.name,
        search_player,
        length(initialstates),
    )
    flush(stdout)
    Random.seed!(seed + search_player)
    elapsed = @elapsed result = Tools.evaluate_joint_policy(
        game,
        joint_policy,
        length(initialstates);
        max_steps,
        initialstates,
        show_progress=false,
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
    reward = result.reward[search_player]
    stderr = result.stderr_reward[search_player]
    @printf(
        "[scheme-screen] done candidate=%s player=%d reward=%.6f stderr=%.6f elapsed=%.1fs\n",
        config.name,
        search_player,
        reward,
        stderr,
        elapsed,
    )
    flush(stdout)
    return merge(config, (;
        search_player,
        runs=length(initialstates),
        reward,
        stderr_reward=stderr,
        mean_steps=result.mean_steps,
        detection_rate=result.detected_rate,
        target_escaped_rate=result.target_escaped_rate,
        observer_lost_rate=result.observer_lost_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        elapsed_seconds=elapsed,
    ))
end

function csv_value(value)
    value isa AbstractString && return value
    value isa Symbol && return String(value)
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

function selected_candidates(all_candidates, names)
    isempty(names) && return all_candidates
    requested = Set(split(names, ','))
    selected = filter(config -> config.name in requested, all_candidates)
    found = Set(config.name for config in selected)
    missing = setdiff(requested, found)
    isempty(missing) || error("Unknown candidates: $(join(sort!(collect(missing)), ", "))")
    return selected
end

function main(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 48, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 100, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260722, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    only_names = option_value(args, "--only", "", String)
    output_dir = abspath(option_value(args, "--output-dir", DEFAULT_OUTPUT_DIR, String))
    fitted_models_path = abspath(option_value(
        args,
        "--fitted-models",
        DEFAULT_FITTED_MODELS,
        String,
    ))
    if test
        runs = min(runs, 2)
        queries = min(queries, 2)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
        only_names = "value_only,baseline_raw_s1.0"
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    max_steps > 0 || error("--max-steps must be positive")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")
    isfile(fitted_models_path) || error("Missing fitted models $(fitted_models_path)")

    checkpoint = select_checkpoint(checkpoint_spec)
    oracle_file = joinpath(@__DIR__, "oracle_$(SEARCH_NAME).jld2")
    online_oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(online_oracle, checkpoint)
    baseline_oracle, hurdle_regressors, metadata =
        build_refit_oracles(online_oracle, fitted_models_path)

    game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
    initial_rng = MersenneTwister(seed)
    initial_distribution = core_initialstate_distribution(game)
    initialstates = [rand(initial_rng, initial_distribution) for _ in 1:runs]
    candidates = selected_candidates(screening_candidates(), only_names)

    println("[scheme-screen] checkpoint=$(checkpoint)")
    println("[scheme-screen] fitted_models=$(fitted_models_path)")
    baseline_activation = metadata["baseline_activation"]
    fit_epochs = metadata["epochs"]
    println(
        "[scheme-screen] baseline_activation=$(baseline_activation) ",
        "fit_epochs=$(fit_epochs) candidates=$(length(candidates))",
    )
    println(
        "[scheme-screen] runs=$(runs) queries=$(queries) depth=$(max_depth) ",
        "steps=$(max_steps) search_epsilon=$(search_epsilon) action_epsilon=0.0",
    )
    flush(stdout)

    rows = NamedTuple[]
    for config in candidates, search_player in 1:2
        push!(rows, evaluate_candidate(
            game,
            initialstates,
            baseline_oracle,
            hurdle_regressors,
            config,
            search_player;
            max_steps,
            queries,
            max_depth,
            search_epsilon,
            seed,
        ))
    end
    value_only = Dict(
        row.search_player => row.reward for row in rows if row.name == "value_only"
    )
    if length(value_only) == 2
        rows = map(rows) do row
            merge(row, (delta_vs_value_only=row.reward - value_only[row.search_player],))
        end
    end

    mkpath(output_dir)
    output = write_csv(joinpath(output_dir, "heuristic_matchups.csv"), rows)
    println("[scheme-screen] wrote $(output)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
