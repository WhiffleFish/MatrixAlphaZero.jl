using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DelimitedFiles
using Distributions
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const EXPERIMENT_DIR = @__DIR__
const SEARCH_NAME = "rm_plus_transfer"
const CLEANRL_TREE_SHA = "f23b4c0783c380ab8337c244dbb2182e60e63387"
const SOLVERS = ("zero_oracle", "value_oracle", "full_solver")
const MAX_PPO_STEPS = 50

# PPO is trained once for each player against three otherwise-identical finite-
# depth RM+ planners. These are empirical response utilities, not exact best
# responses and not an exact exploitability calculation.
#
#   zero_oracle  : V(s)=0, uniform fallback policy, no transfer
#   value_oracle : learned V(s), uniform fallback policy, no transfer
#   full_solver  : learned V(s), regret transfer, and strategy transfer

struct ValueOnlySearchOracle{O}
    oracle::O
    na::NTuple{2,Int}
end

ValueOnlySearchOracle(game::MG, oracle) =
    ValueOnlySearchOracle(oracle, Tuple(length.(actions(game))))

uniform_pair(oracle::ValueOnlySearchOracle) = (
    fill(Float32(inv(oracle.na[1])), oracle.na[1]),
    fill(Float32(inv(oracle.na[2])), oracle.na[2]),
)

AZ.value(oracle::ValueOnlySearchOracle, x) = AZ.value(oracle.oracle, x)
AZ.state_value(oracle::ValueOnlySearchOracle, game, s) =
    AZ.state_value(oracle.oracle, game, s)
AZ.batch_state_value(oracle::ValueOnlySearchOracle, game, states) =
    AZ.batch_state_value(oracle.oracle, game, states)
AZ.state_policy(oracle::ValueOnlySearchOracle, game, s) = uniform_pair(oracle)
AZ.batch_state_policy(oracle::ValueOnlySearchOracle, game, states) = (
    fill(Float32(inv(oracle.na[1])), oracle.na[1], length(states)),
    fill(Float32(inv(oracle.na[2])), oracle.na[2], length(states)),
)
AZ.state_strategy(oracle::ValueOnlySearchOracle, game, s) = uniform_pair(oracle)
AZ.batch_state_strategy(oracle::ValueOnlySearchOracle, game, states) =
    AZ.batch_state_policy(oracle, game, states)
AZ.state_regret(oracle::ValueOnlySearchOracle, game, s) = (
    zeros(Float32, oracle.na[1]),
    zeros(Float32, oracle.na[2]),
)
AZ.batch_state_regret(oracle::ValueOnlySearchOracle, game, states) = (
    zeros(Float32, oracle.na[1], length(states)),
    zeros(Float32, oracle.na[2], length(states)),
)

function make_game()
    d_observer = ImplicitDistribution() do rng
        SNRGame.sOSCtoCART2D([
            R_EARTH + rand(rng, Distributions.Uniform(500e3, 1e7)),
            0.0,
            0.0,
            rand(rng) * 2π,
        ])
    end
    d_target = ImplicitDistribution() do rng
        SNRGame.sOSCtoCART2D([
            R_EARTH + rand(rng, Distributions.Uniform(500e3, 1e7)),
            0.0,
            0.0,
            rand(rng) * 2π,
        ])
    end
    return SNRGameSimple(
        observer=d_observer,
        target=d_target,
        altitude_bounds=(100e3, 2e7),
    )
end

split_nonempty(s::AbstractString) =
    String.(filter(!isempty, strip.(split(s, ","))))

function parse_players(spec::AbstractString)
    spec == "both" && return [1, 2]
    return map(split_nonempty(spec)) do item
        item in ("1", "p1", "P1") && return 1
        item in ("2", "p2", "P2") && return 2
        error("Unknown player spec $(item)")
    end
end

function parse_args(args)
    cfg = Dict{String,String}(
        "solvers" => join(SOLVERS, ","),
        "players" => "both",
        "iter" => "latest",
        "backup" => "sample",
        "value-target" => "search",
        "total-timesteps" => "500000",
        "num-steps" => "128",
        "num-envs" => "4",
        "num-minibatches" => "4",
        "update-epochs" => "4",
        "lr" => "2.5e-4",
        "gamma" => "0.98",
        "gae-lambda" => "0.95",
        "clip-coef" => "0.2",
        "ent-coeff" => "0.01",
        "v-coef" => "0.5",
        "anneal-lr" => "true",
        "normalize-advantages" => "true",
        "clip-value-loss" => "true",
        "max-steps" => string(MAX_PPO_STEPS),
        "eval-runs" => "100",
        "tree-queries" => "500",
        "max-depth" => "5",
        "epsilon" => "0.0",
        "source-mass" => "auto",
        "train-tree-queries" => "500",
        "regret-confidence" => "0.11827004532587074",
        "strategy-confidence" => "0.39183430645643386",
        "transfer-payoff-bound" => "Inf",
        "regret-scale" => "0.25",
        "strategy-scale" => "0.25",
        "reach-power" => "1.0",
        "seed" => "20260719",
        "output-dir" => "auto",
        "fail-fast" => "false",
        "test" => "false",
    )

    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        opt = key[3:end]
        haskey(cfg, opt) || error("Unknown option $(key)")
        if opt == "test"
            cfg[opt] = "true"
            i += 1
            continue
        end
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        cfg[opt] = args[i]
        i += 1
    end

    output_dir = cfg["output-dir"] == "auto" ?
        joinpath(EXPERIMENT_DIR, "ppo_solver_response_utility_results") :
        abspath(cfg["output-dir"])
    solvers = split_nonempty(cfg["solvers"])
    unknown = setdiff(solvers, collect(SOLVERS))
    isempty(unknown) || error(
        "Unknown solvers $(unknown); expected $(join(SOLVERS, ", "))",
    )

    parsed = (;
        solvers,
        players=parse_players(cfg["players"]),
        iter=cfg["iter"],
        backup=Symbol(cfg["backup"]),
        value_target=Symbol(cfg["value-target"]),
        total_timesteps=parse(Int, cfg["total-timesteps"]),
        num_steps=parse(Int, cfg["num-steps"]),
        num_envs=parse(Int, cfg["num-envs"]),
        num_minibatches=parse(Int, cfg["num-minibatches"]),
        update_epochs=parse(Int, cfg["update-epochs"]),
        lr=parse(Float32, cfg["lr"]),
        gamma=parse(Float32, cfg["gamma"]),
        gae_lambda=parse(Float32, cfg["gae-lambda"]),
        clip_coef=parse(Float32, cfg["clip-coef"]),
        ent_coeff=parse(Float32, cfg["ent-coeff"]),
        v_coef=parse(Float32, cfg["v-coef"]),
        anneal_lr=parse(Bool, cfg["anneal-lr"]),
        normalize_advantages=parse(Bool, cfg["normalize-advantages"]),
        clip_value_loss=parse(Bool, cfg["clip-value-loss"]),
        max_steps=parse(Int, cfg["max-steps"]),
        eval_runs=parse(Int, cfg["eval-runs"]),
        tree_queries=parse(Int, cfg["tree-queries"]),
        max_depth=parse(Int, cfg["max-depth"]),
        epsilon=parse(Float64, cfg["epsilon"]),
        source_mass=cfg["source-mass"] == "auto" ?
            nothing : parse(Float64, cfg["source-mass"]),
        train_tree_queries=parse(Int, cfg["train-tree-queries"]),
        regret_confidence=parse(Float64, cfg["regret-confidence"]),
        strategy_confidence=parse(Float64, cfg["strategy-confidence"]),
        transfer_payoff_bound=parse(Float64, cfg["transfer-payoff-bound"]),
        regret_scale=parse(Float64, cfg["regret-scale"]),
        strategy_scale=parse(Float64, cfg["strategy-scale"]),
        reach_power=parse(Float64, cfg["reach-power"]),
        seed=parse(Int, cfg["seed"]),
        output_dir,
        fail_fast=parse(Bool, cfg["fail-fast"]),
        test=parse(Bool, cfg["test"]),
    )
    validate_config(parsed)
    return parsed.test ? test_config(parsed) : parsed
end

function validate_config(cfg)
    cfg.backup in (:sample, :mean) || error("--backup must be sample or mean")
    cfg.value_target in (:search, :gae) ||
        error("--value-target must be search or gae")
    cfg.tree_queries >= 0 || error("--tree-queries must be nonnegative")
    cfg.max_depth >= 0 || error("--max-depth must be nonnegative")
    0 <= cfg.epsilon <= 1 || error("--epsilon must be in [0, 1]")
    isnothing(cfg.source_mass) || cfg.source_mass >= 0 ||
        error("--source-mass must be auto or nonnegative")
    cfg.train_tree_queries > 0 || error("--train-tree-queries must be positive")
    0 <= cfg.regret_confidence <= 1 ||
        error("--regret-confidence must be in [0, 1]")
    0 <= cfg.strategy_confidence <= 1 ||
        error("--strategy-confidence must be in [0, 1]")
    cfg.transfer_payoff_bound > 0 ||
        error("--transfer-payoff-bound must be positive")
    cfg.regret_scale >= 0 || error("--regret-scale must be nonnegative")
    cfg.strategy_scale >= 0 || error("--strategy-scale must be nonnegative")
    cfg.reach_power >= 0 || error("--reach-power must be nonnegative")
    0 < cfg.max_steps <= MAX_PPO_STEPS || error(
        "--max-steps must be in 1:$(MAX_PPO_STEPS); this harness does not increase PPO's horizon above Dubin",
    )
    batch_size = cfg.num_steps * cfg.num_envs
    cfg.total_timesteps >= batch_size || error(
        "--total-timesteps must be at least num-steps × num-envs = $(batch_size)",
    )
    iszero(batch_size % cfg.num_minibatches) || error(
        "num-steps × num-envs must be divisible by --num-minibatches",
    )
    return cfg
end

function test_config(cfg)
    # Keep max_steps unchanged: the smoke tests the same 50-step PPO wrapper.
    return merge(cfg, (;
        total_timesteps=64,
        num_steps=8,
        num_envs=1,
        num_minibatches=1,
        update_epochs=1,
        eval_runs=2,
        tree_queries=2,
        max_depth=2,
        output_dir=joinpath(cfg.output_dir, "smoke"),
        fail_fast=true,
    ))
end

function checkpoint_iteration(path::AbstractString)
    m = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(m) && error("Cannot parse checkpoint iteration from $(path)")
    return parse(Int, m.captures[1])
end

function checkpoint_paths()
    models_dir = joinpath(EXPERIMENT_DIR, "models_$(SEARCH_NAME)")
    isdir(models_dir) || error("Missing model checkpoint directory: $(models_dir)")
    checkpoints = filter(
        p -> occursin(r"oracle\d+\.jld2$", basename(p)),
        readdir(models_dir; join=true),
    )
    isempty(checkpoints) && error("No oracle checkpoints found in $(models_dir)")
    sort!(checkpoints; by=checkpoint_iteration)
    return checkpoints
end

function select_checkpoint(iter_spec::AbstractString)
    checkpoints = checkpoint_paths()
    iter_spec == "latest" && return last(checkpoints)
    iter = parse(Int, iter_spec)
    matches = filter(p -> checkpoint_iteration(p) == iter, checkpoints)
    isempty(matches) && error("No checkpoint for iteration $(iter)")
    return only(matches)
end

function load_checkpoint_oracle(iter_spec::AbstractString)
    oracle_file = joinpath(EXPERIMENT_DIR, "oracle_$(SEARCH_NAME).jld2")
    isfile(oracle_file) || error("Missing oracle architecture file: $(oracle_file)")
    checkpoint = select_checkpoint(iter_spec)
    oracle = AZ.load_oracle(oracle_file)
    oracle isa AZ.FittedRegretModel || error(
        "Full solver requires a FittedRegretModel, got $(typeof(oracle))",
    )
    Flux.loadmodel!(oracle, checkpoint)
    return oracle, checkpoint_iteration(checkpoint), checkpoint
end

function base_search(oracle, cfg)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        search_style=AZ.RegretMatchingSearch(;
            backup=cfg.backup,
            method=AZ.Plus(),
        ),
        value_target=cfg.value_target,
        ϵ=_ -> cfg.epsilon,
    )
end

function full_search(oracle, cfg)
    transfer = AZ.LossScaledTransfer(;
        regret_scale=cfg.regret_scale,
        strategy_scale=cfg.strategy_scale,
        reach_power=cfg.reach_power,
    )
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        search_style=AZ.RegretMatchingSearch(;
            backup=cfg.backup,
            method=AZ.Plus(),
        ),
        value_target=cfg.value_target,
        ϵ=_ -> cfg.epsilon,
        τ=cfg.source_mass,
        transfer_weight=0.0,
        transfer_payoff_bound=cfg.transfer_payoff_bound,
        loss_scaled_transfer=transfer,
        regret_confidence=cfg.regret_confidence,
        strategy_confidence=cfg.strategy_confidence,
    )
end

function solver_policy(solver::AbstractString, game, learned_oracle, cfg)
    if solver == "zero_oracle"
        search = base_search(Tools.ZeroSearchOracle(game), cfg)
        meta = (;
            oracle_kind="uniform_zero",
            uses_value_oracle=false,
            uses_transfer=false,
            regret_mass=0.0,
            strategy_mass=0.0,
        )
    elseif solver == "value_oracle"
        search = base_search(ValueOnlySearchOracle(game, learned_oracle), cfg)
        meta = (;
            oracle_kind="learned_value_only",
            uses_value_oracle=true,
            uses_transfer=false,
            regret_mass=0.0,
            strategy_mass=0.0,
        )
    elseif solver == "full_solver"
        search = full_search(learned_oracle, cfg)
        regret_mass, strategy_mass = AZ.transfer_pseudo_masses(search)
        meta = (;
            oracle_kind="learned_value_regret_strategy",
            uses_value_oracle=true,
            uses_transfer=true,
            regret_mass,
            strategy_mass,
        )
    else
        error("Unsupported solver $(solver)")
    end
    return AZ.AlphaZeroPlanner(game, search), meta
end

function ppo_log_name(solver::AbstractString, br_player::Int)
    fixed_player = MarkovGames.other_player(br_player)
    return "response_p$(br_player)__vs_$(solver)_p$(fixed_player)"
end

function ppo_config(cfg, seed::Int, name::String)
    return Tools.PPOBestResponseConfig(;
        total_timesteps=cfg.total_timesteps,
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        num_minibatches=cfg.num_minibatches,
        update_epochs=cfg.update_epochs,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_coef=cfg.clip_coef,
        ent_coeff=cfg.ent_coeff,
        v_coef=cfg.v_coef,
        normalize_advantages=cfg.normalize_advantages,
        clip_value_loss=cfg.clip_value_loss,
        anneal_lr=cfg.anneal_lr,
        max_steps=cfg.max_steps,
        seed,
        name,
        log_dir=joinpath(cfg.output_dir, "tensorboard"),
    )
end

function rollout_eval(game, joint_policy, initialstates; runs::Int, max_steps::Int)
    return Tools.evaluate_joint_policy(
        game,
        joint_policy,
        runs;
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
end

evaluation_initialstates(rng, dist, n::Int) = [rand(rng, dist) for _ in 1:n]

function run_one(solver::String, br_player::Int, game, oracle, model_iter, checkpoint, cfg)
    fixed, solver_meta = solver_policy(solver, game, oracle, cfg)
    init_dist = initialstate(game)
    # Pair PPO initialization and evaluation states across solver conditions.
    local_seed = cfg.seed + 10_000 * br_player
    ppo_name = ppo_log_name(solver, br_player)

    println("Training PPO response: solver=$(solver) player=$(br_player) name=$(ppo_name)")
    response = Tools.train_ppo_best_response(
        game,
        fixed,
        br_player;
        initialstate_dist=init_dist,
        config=ppo_config(cfg, local_seed, ppo_name),
    )
    actor, critic = response.actor, response.critic

    eval_rng = MersenneTwister(local_seed + 1)
    eval_states = evaluation_initialstates(eval_rng, init_dist, cfg.eval_runs)
    joint_policy = Tools.ppo_best_response_joint_policy(
        game,
        fixed,
        actor,
        br_player,
    )
    eval_result = rollout_eval(
        game,
        joint_policy,
        eval_states;
        runs=cfg.eval_runs,
        max_steps=cfg.max_steps,
    )

    response_reward = eval_result.reward[br_player]
    response_stderr = eval_result.stderr_reward[br_player]
    all(isfinite, (response_reward, response_stderr, eval_result.reward...)) ||
        error("Non-finite PPO evaluation for solver=$(solver), p$(br_player)")

    solver_dir = joinpath(cfg.output_dir, solver, "p$(br_player)")
    mkpath(solver_dir)
    model_path = joinpath(solver_dir, "ppo_response_actor_critic.jld2")
    metadata = Dict(
        "solver" => solver,
        "response_player" => br_player,
        "seed" => local_seed,
        "model_iter" => model_iter,
        "checkpoint" => checkpoint,
        "cleanrl_tree_sha" => CLEANRL_TREE_SHA,
        "ppo_name" => ppo_name,
        "total_timesteps" => cfg.total_timesteps,
        "num_envs" => cfg.num_envs,
        "num_steps" => cfg.num_steps,
        "max_steps" => cfg.max_steps,
        "gamma" => cfg.gamma,
        "tree_queries" => cfg.tree_queries,
        "max_depth" => cfg.max_depth,
        "epsilon" => cfg.epsilon,
        "source_mass" => cfg.source_mass,
        "regret_confidence" => cfg.regret_confidence,
        "strategy_confidence" => cfg.strategy_confidence,
        "regret_mass" => solver_meta.regret_mass,
        "strategy_mass" => solver_meta.strategy_mass,
        "initial_state" => "game_distribution",
    )
    jldsave(model_path; actor, critic, metadata)

    println(
        "eval solver=$(solver) p$(br_player) response reward=",
        "$(round(response_reward; digits=4)) stderr=",
        "$(round(response_stderr; digits=4)) steps=",
        "$(round(eval_result.mean_steps; digits=2)) detection=",
        "$(round(eval_result.detected_rate; digits=3))",
    )

    return (;
        solver,
        solver_meta.oracle_kind,
        solver_meta.uses_value_oracle,
        solver_meta.uses_transfer,
        response_player=br_player,
        fixed_player=MarkovGames.other_player(br_player),
        seed=local_seed,
        model_iter,
        checkpoint,
        cleanrl_tree_sha=CLEANRL_TREE_SHA,
        ppo_name,
        total_timesteps=cfg.total_timesteps,
        num_envs=cfg.num_envs,
        num_steps=cfg.num_steps,
        max_steps=cfg.max_steps,
        gamma=cfg.gamma,
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        epsilon=cfg.epsilon,
        source_mass=cfg.source_mass,
        regret_confidence=cfg.regret_confidence,
        strategy_confidence=cfg.strategy_confidence,
        regret_mass=solver_meta.regret_mass,
        strategy_mass=solver_meta.strategy_mass,
        initial_state="game_distribution",
        response_reward,
        response_stderr_reward=response_stderr,
        p1_reward=eval_result.reward[1],
        p2_reward=eval_result.reward[2],
        detection_rate=eval_result.detected_rate,
        target_escaped_rate=eval_result.target_escaped_rate,
        observer_lost_rate=eval_result.observer_lost_rate,
        mean_steps=eval_result.mean_steps,
        model_path,
    )
end

const RESULT_COLUMNS = (
    :solver,
    :oracle_kind,
    :uses_value_oracle,
    :uses_transfer,
    :response_player,
    :fixed_player,
    :seed,
    :model_iter,
    :checkpoint,
    :cleanrl_tree_sha,
    :ppo_name,
    :total_timesteps,
    :num_envs,
    :num_steps,
    :max_steps,
    :gamma,
    :tree_queries,
    :max_depth,
    :epsilon,
    :source_mass,
    :regret_confidence,
    :strategy_confidence,
    :regret_mass,
    :strategy_mass,
    :initial_state,
    :response_reward,
    :response_stderr_reward,
    :p1_reward,
    :p2_reward,
    :detection_rate,
    :target_escaped_rate,
    :observer_lost_rate,
    :mean_steps,
    :model_path,
)

function write_detailed_summary(path, results)
    header = reshape(collect(String.(RESULT_COLUMNS)), 1, :)
    data = isempty(results) ? Matrix{Any}(undef, 0, length(RESULT_COLUMNS)) :
        reduce(vcat, [
            reshape(
                Any[getproperty(result, key) for key in RESULT_COLUMNS],
                1,
                :,
            ) for result in results
        ])
    mkpath(dirname(path))
    writedlm(path, [header; data], ',')
    return path
end

function response_summary_rows(results, solvers)
    rows = NamedTuple[]
    for solver in solvers
        solver_results = filter(r -> r.solver == solver, results)
        p1 = filter(r -> r.response_player == 1, solver_results)
        p2 = filter(r -> r.response_player == 2, solver_results)
        if length(p1) != 1 || length(p2) != 1
            @warn "Skipping response-utility sum for incomplete solver" solver
            continue
        end
        r1, r2 = only(p1), only(p2)
        utility_sum = r1.response_reward + r2.response_reward
        utility_sum_stderr = hypot(
            r1.response_stderr_reward,
            r2.response_stderr_reward,
        )
        informative_nonnegative = utility_sum >= 0
        informative_nonnegative || @warn(
            "Negative summed PPO response utility is an uninformative lower bound; at least one response policy underfit",
            solver,
            utility_sum,
            utility_sum_stderr,
        )
        push!(rows, (;
            solver,
            response_p1_reward=r1.response_reward,
            response_p1_stderr=r1.response_stderr_reward,
            response_p2_reward=r2.response_reward,
            response_p2_stderr=r2.response_stderr_reward,
            summed_ppo_response_utility=utility_sum,
            summed_ppo_response_utility_stderr=utility_sum_stderr,
            half_summed_ppo_response_utility=0.5 * utility_sum,
            half_summed_ppo_response_utility_stderr=0.5 * utility_sum_stderr,
            informative_nonnegative,
        ))
    end
    return rows
end

function write_response_summary(path, rows)
    columns = (
        :solver,
        :response_p1_reward,
        :response_p1_stderr,
        :response_p2_reward,
        :response_p2_stderr,
        :summed_ppo_response_utility,
        :summed_ppo_response_utility_stderr,
        :half_summed_ppo_response_utility,
        :half_summed_ppo_response_utility_stderr,
        :informative_nonnegative,
    )
    header = reshape(collect(String.(columns)), 1, :)
    data = isempty(rows) ? Matrix{Any}(undef, 0, length(columns)) :
        reduce(vcat, [
            reshape(Any[getproperty(row, key) for key in columns], 1, :)
            for row in rows
        ])
    writedlm(path, [header; data], ',')
    return path
end

function write_failures(path, failures)
    isempty(failures) && return nothing
    rows = Any["solver" "response_player" "error"]
    for failure in failures
        rows = vcat(
            rows,
            Any[failure.solver failure.response_player failure.error],
        )
    end
    writedlm(path, rows, ',')
    return path
end

function main(args)
    cfg = parse_args(args)
    game = make_game()
    oracle, model_iter, checkpoint = load_checkpoint_oracle(cfg.iter)
    cfg = isnothing(cfg.source_mass) ?
        merge(cfg, (; source_mass=Float64(model_iter * cfg.train_tree_queries))) :
        cfg
    mkpath(cfg.output_dir)
    println("Loaded checkpoint: $(checkpoint)")
    println("Solvers: $(join(cfg.solvers, ", "))")
    println("Response players: $(join(cfg.players, ", "))")
    println(
        "PPO horizon=$(cfg.max_steps) gamma=$(cfg.gamma); search epsilon=$(cfg.epsilon)",
    )
    println(
        "Transfer state: source_mass=$(cfg.source_mass) ",
        "regret_confidence=$(cfg.regret_confidence) ",
        "strategy_confidence=$(cfg.strategy_confidence)",
    )

    results = NamedTuple[]
    failures = NamedTuple[]
    for solver in cfg.solvers, response_player in cfg.players
        try
            push!(results, run_one(
                solver,
                response_player,
                game,
                oracle,
                model_iter,
                checkpoint,
                cfg,
            ))
        catch err
            cfg.fail_fast && rethrow()
            message = replace(sprint(showerror, err), '\n' => ' ')
            @error(
                "PPO response-utility evaluation failed",
                solver,
                response_player,
                exception=(err, catch_backtrace()),
            )
            push!(failures, (; solver, response_player, error=message))
        end
    end

    detailed_path = write_detailed_summary(
        joinpath(cfg.output_dir, "best_response_utilities.csv"),
        results,
    )
    aggregate = response_summary_rows(results, cfg.solvers)
    summary_path = write_response_summary(
        joinpath(cfg.output_dir, "response_utility_summary.csv"),
        aggregate,
    )
    failure_path = write_failures(
        joinpath(cfg.output_dir, "failures.csv"),
        failures,
    )

    println("Wrote PPO response utilities: $(detailed_path)")
    println("Wrote response-utility summary: $(summary_path)")
    isnothing(failure_path) || println("Wrote failures: $(failure_path)")
    return (; results, aggregate, failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
