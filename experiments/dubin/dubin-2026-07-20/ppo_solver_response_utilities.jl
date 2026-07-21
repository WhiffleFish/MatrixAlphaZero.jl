using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DelimitedFiles
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_DIR = @__DIR__
const SEARCH_NAME = "rm_plus_no_transfer_train"
const CLEANRL_TREE_SHA = "f23b4c0783c380ab8337c244dbb2182e60e63387"
const SOLVERS = ("zero_oracle", "value_oracle", "full_solver")
const MAX_PPO_STEPS = 50

# PPO is trained once for each player against three otherwise-identical
# finite-depth RM+ planners. These are empirical response utilities, not exact
# best responses or exact exploitability values.
#
#   zero_oracle  : V(s)=0, uniform priors, no warm start
#   value_oracle : learned V(s), uniform priors, no warm start
#   full_solver  : learned V(s), regret/strategy/count/value warm start

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
    raw = Dict{String,String}(
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
        "gamma" => "0.99",
        "gae-lambda" => "0.95",
        "clip-coef" => "0.2",
        "ent-coeff" => "0.01",
        "v-coef" => "0.5",
        "anneal-lr" => "true",
        "normalize-advantages" => "true",
        "clip-value-loss" => "true",
        "max-steps" => string(MAX_PPO_STEPS),
        "eval-runs" => "100",
        "tree-queries" => "100",
        "max-depth" => "5",
        "epsilon" => "0.1",
        "prior-scale" => "100.0",
        "seed" => "20260720",
        "initial-state" => "reference",
        "output-dir" => "auto",
        "fail-fast" => "false",
        "test" => "false",
    )

    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        option = key[3:end]
        haskey(raw, option) || error("Unknown option $(key)")
        if option == "test"
            raw[option] = "true"
            i += 1
            continue
        end
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        raw[option] = args[i]
        i += 1
    end

    output_dir = raw["output-dir"] == "auto" ?
        joinpath(EXPERIMENT_DIR, "ppo_solver_response_utility_results") :
        abspath(raw["output-dir"])
    solvers = split_nonempty(raw["solvers"])
    unknown = setdiff(solvers, collect(SOLVERS))
    isempty(unknown) || error(
        "Unknown solvers $(unknown); expected $(join(SOLVERS, ", "))",
    )

    cfg = (;
        solvers,
        players=parse_players(raw["players"]),
        iter=raw["iter"],
        backup=Symbol(raw["backup"]),
        value_target=Symbol(raw["value-target"]),
        total_timesteps=parse(Int, raw["total-timesteps"]),
        num_steps=parse(Int, raw["num-steps"]),
        num_envs=parse(Int, raw["num-envs"]),
        num_minibatches=parse(Int, raw["num-minibatches"]),
        update_epochs=parse(Int, raw["update-epochs"]),
        lr=parse(Float32, raw["lr"]),
        gamma=parse(Float32, raw["gamma"]),
        gae_lambda=parse(Float32, raw["gae-lambda"]),
        clip_coef=parse(Float32, raw["clip-coef"]),
        ent_coeff=parse(Float32, raw["ent-coeff"]),
        v_coef=parse(Float32, raw["v-coef"]),
        anneal_lr=parse(Bool, raw["anneal-lr"]),
        normalize_advantages=parse(Bool, raw["normalize-advantages"]),
        clip_value_loss=parse(Bool, raw["clip-value-loss"]),
        max_steps=parse(Int, raw["max-steps"]),
        eval_runs=parse(Int, raw["eval-runs"]),
        tree_queries=parse(Int, raw["tree-queries"]),
        max_depth=parse(Int, raw["max-depth"]),
        epsilon=parse(Float64, raw["epsilon"]),
        prior_scale=parse(Float64, raw["prior-scale"]),
        seed=parse(Int, raw["seed"]),
        initial_state=raw["initial-state"],
        output_dir,
        fail_fast=parse(Bool, raw["fail-fast"]),
        test=parse(Bool, raw["test"]),
    )
    validate_config(cfg)
    return cfg.test ? test_config(cfg) : cfg
end

function validate_config(cfg)
    cfg.backup in (:sample, :mean) || error("--backup must be sample or mean")
    cfg.value_target in (:search, :gae) ||
        error("--value-target must be search or gae")
    cfg.tree_queries >= 0 || error("--tree-queries must be nonnegative")
    cfg.max_depth >= 0 || error("--max-depth must be nonnegative")
    0 <= cfg.epsilon <= 1 || error("--epsilon must be in [0, 1]")
    cfg.prior_scale >= 0 || error("--prior-scale must be nonnegative")
    0 < cfg.max_steps <= MAX_PPO_STEPS || error(
        "--max-steps must be in 1:$(MAX_PPO_STEPS)",
    )
    cfg.initial_state in ("reference", "game") ||
        error("--initial-state must be reference or game")
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
    # Preserve the 50-step episode horizon in the smoke test.
    return merge(cfg, (;
        total_timesteps=64,
        num_steps=8,
        num_envs=1,
        num_minibatches=1,
        update_epochs=1,
        eval_runs=2,
        tree_queries=2,
        max_depth=2,
        prior_scale=min(cfg.prior_scale, 2.0),
        output_dir=joinpath(cfg.output_dir, "smoke"),
        fail_fast=true,
    ))
end

function checkpoint_iteration(path::AbstractString)
    match_result = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(match_result) && error("Cannot parse checkpoint iteration from $(path)")
    return parse(Int, match_result.captures[1])
end

function checkpoint_paths()
    models_dir = joinpath(EXPERIMENT_DIR, "models_$(SEARCH_NAME)")
    isdir(models_dir) || error("Missing model checkpoint directory: $(models_dir)")
    checkpoints = filter(
        path -> occursin(r"oracle\d+\.jld2$", basename(path)),
        readdir(models_dir; join=true),
    )
    isempty(checkpoints) && error("No oracle checkpoints found in $(models_dir)")
    sort!(checkpoints; by=checkpoint_iteration)
    return checkpoints
end

function select_checkpoint(iter_spec::AbstractString)
    checkpoints = checkpoint_paths()
    iter_spec == "latest" && return last(checkpoints)
    iteration = parse(Int, iter_spec)
    matches = filter(path -> checkpoint_iteration(path) == iteration, checkpoints)
    isempty(matches) && error("No checkpoint for iteration $(iteration)")
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

function initial_dubin_state()
    return JointDubinState(
        SA[1, 1, deg2rad(45)],
        SA[8, 7, deg2rad(180)],
    )
end

function initialstate_dist(game, spec::AbstractString)
    spec == "reference" && return Deterministic(initial_dubin_state())
    spec == "game" && return initialstate(game)
    error("Unsupported initial-state specification $(spec)")
end

function build_search(oracle, cfg; prior_scale::Real=0.0)
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
        prior_scale=Float64(prior_scale),
    )
end

function solver_policy(solver::AbstractString, game, learned_oracle, cfg)
    if solver == "zero_oracle"
        oracle = Tools.ZeroSearchOracle(game)
        search = build_search(oracle, cfg)
        meta = (;
            oracle_kind="uniform_zero",
            uses_value_oracle=false,
            uses_transfer=false,
            prior_scale=0.0,
        )
    elseif solver == "value_oracle"
        oracle = ValueOnlySearchOracle(game, learned_oracle)
        search = build_search(oracle, cfg)
        meta = (;
            oracle_kind="learned_value_only",
            uses_value_oracle=true,
            uses_transfer=false,
            prior_scale=0.0,
        )
    elseif solver == "full_solver"
        search = build_search(learned_oracle, cfg; prior_scale=cfg.prior_scale)
        meta = (;
            oracle_kind="learned_value_regret_strategy",
            uses_value_oracle=true,
            uses_transfer=!iszero(cfg.prior_scale),
            prior_scale=cfg.prior_scale,
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
        accumulators=(StepCount(), DubinTools.DubinOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(
                :reward;
                name=:stderr_reward,
                init=zero(MarkovGames.reward_type(game)),
            ),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
    )
end

evaluation_initialstates(rng, dist, n::Int) = [rand(rng, dist) for _ in 1:n]

function run_one(solver::String, br_player::Int, game, oracle, model_iter, checkpoint, cfg)
    fixed, solver_meta = solver_policy(solver, game, oracle, cfg)
    init_dist = initialstate_dist(game, cfg.initial_state)
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
    joint_policy = Tools.ppo_best_response_joint_policy(game, fixed, actor, br_player)
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
        "prior_scale" => solver_meta.prior_scale,
        "initial_state" => cfg.initial_state,
    )
    jldsave(model_path; actor, critic, metadata)

    println(
        "eval solver=$(solver) p$(br_player) response reward=",
        "$(round(response_reward; digits=4)) stderr=",
        "$(round(response_stderr; digits=4)) steps=",
        "$(round(eval_result.mean_steps; digits=2)) goal=",
        "$(round(eval_result.attacker_goal_rate; digits=3)) tagged=",
        "$(round(eval_result.tagged_rate; digits=3))",
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
        prior_scale=solver_meta.prior_scale,
        initial_state=cfg.initial_state,
        response_reward,
        response_stderr_reward=response_stderr,
        p1_reward=eval_result.reward[1],
        p2_reward=eval_result.reward[2],
        attacker_goal_rate=eval_result.attacker_goal_rate,
        tagged_rate=eval_result.tagged_rate,
        timeout_rate=eval_result.timeout_rate,
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
    :prior_scale,
    :initial_state,
    :response_reward,
    :response_stderr_reward,
    :p1_reward,
    :p2_reward,
    :attacker_goal_rate,
    :tagged_rate,
    :timeout_rate,
    :mean_steps,
    :model_path,
)

function write_named_tuples(path, rows, columns)
    header = reshape(collect(String.(columns)), 1, :)
    data = isempty(rows) ? Matrix{Any}(undef, 0, length(columns)) : reduce(
        vcat,
        [reshape(Any[getproperty(row, key) for key in columns], 1, :) for row in rows],
    )
    mkpath(dirname(path))
    writedlm(path, [header; data], ',')
    return path
end

function response_summary_rows(results, solvers)
    rows = NamedTuple[]
    for solver in solvers
        solver_results = filter(result -> result.solver == solver, results)
        p1 = filter(result -> result.response_player == 1, solver_results)
        p2 = filter(result -> result.response_player == 2, solver_results)
        if length(p1) != 1 || length(p2) != 1
            @warn "Skipping incomplete two-player response summary" solver
            continue
        end
        r1, r2 = only(p1), only(p2)
        push!(rows, (;
            solver,
            response_p1_reward=r1.response_reward,
            response_p1_stderr=r1.response_stderr_reward,
            response_p2_reward=r2.response_reward,
            response_p2_stderr=r2.response_stderr_reward,
            response_utility_sum=r1.response_reward + r2.response_reward,
            response_utility_sum_stderr=hypot(
                r1.response_stderr_reward,
                r2.response_stderr_reward,
            ),
        ))
    end
    return rows
end

const SUMMARY_COLUMNS = (
    :solver,
    :response_p1_reward,
    :response_p1_stderr,
    :response_p2_reward,
    :response_p2_stderr,
    :response_utility_sum,
    :response_utility_sum_stderr,
)

function write_failures(path, failures)
    isempty(failures) && return nothing
    rows = Any["solver" "response_player" "error"]
    for failure in failures
        rows = vcat(rows, Any[failure.solver failure.response_player failure.error])
    end
    writedlm(path, rows, ',')
    return path
end

function main(args)
    cfg = parse_args(args)
    game = DubinMG(V=(1.0, 1.0))
    oracle, model_iter, checkpoint = load_checkpoint_oracle(cfg.iter)
    mkpath(cfg.output_dir)
    println("Loaded checkpoint: $(checkpoint)")
    println("Solvers: $(join(cfg.solvers, ", "))")
    println("Players: $(join(cfg.players, ", "))")
    println(
        "Search: queries=$(cfg.tree_queries) depth=$(cfg.max_depth) ",
        "epsilon=$(cfg.epsilon) full_solver_prior_scale=$(cfg.prior_scale)",
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
            @error "PPO response evaluation failed" solver response_player exception=(err, catch_backtrace())
            push!(failures, (; solver, response_player, error=message))
        end
    end

    detailed_path = write_named_tuples(
        joinpath(cfg.output_dir, "response_utilities.csv"),
        results,
        RESULT_COLUMNS,
    )
    summary = response_summary_rows(results, cfg.solvers)
    summary_path = write_named_tuples(
        joinpath(cfg.output_dir, "response_utility_summary.csv"),
        summary,
        SUMMARY_COLUMNS,
    )
    failure_path = write_failures(joinpath(cfg.output_dir, "failures.csv"), failures)

    println("Wrote response utilities: $(detailed_path)")
    println("Wrote response summary: $(summary_path)")
    isnothing(failure_path) || println("Wrote failures: $(failure_path)")
    return (; results, summary, failures)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
