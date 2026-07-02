using Pkg
Pkg.activate("experiments")

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
const CLEANRL_REV = "fd574765ecd9d34eb48fb475403a2add26fe61c6"

function parse_args(args)
    cfg = Dict{String,String}(
        "targets" => "network,oos_value,oos_transfer",
        "players" => "both",
        "iter" => "latest",
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
        "max-steps" => "50",
        "eval-runs" => "100",
        "tree-queries" => "1000",
        "train-oos-iterations" => "1000",
        "max-depth" => "5",
        "epsilon" => "0.3",
        "transfer-weight" => "0.1",
        "seed" => "20260630",
        "initial-state" => "reference",
        "output-dir" => joinpath(EXPERIMENT_DIR, "ppo_best_response_results"),
        "test" => "false",
    )
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        opt = key[3:end]
        haskey(cfg, opt) || error("Unknown option $(key)")
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        cfg[opt] = args[i]
        i += 1
    end
    parsed = (;
        targets = split_nonempty(cfg["targets"]),
        players = parse_players(cfg["players"]),
        iter = cfg["iter"],
        total_timesteps = parse(Int, cfg["total-timesteps"]),
        num_steps = parse(Int, cfg["num-steps"]),
        num_envs = parse(Int, cfg["num-envs"]),
        num_minibatches = parse(Int, cfg["num-minibatches"]),
        update_epochs = parse(Int, cfg["update-epochs"]),
        lr = parse(Float32, cfg["lr"]),
        gamma = parse(Float32, cfg["gamma"]),
        gae_lambda = parse(Float32, cfg["gae-lambda"]),
        clip_coef = parse(Float32, cfg["clip-coef"]),
        ent_coeff = parse(Float32, cfg["ent-coeff"]),
        v_coef = parse(Float32, cfg["v-coef"]),
        anneal_lr = parse(Bool, cfg["anneal-lr"]),
        normalize_advantages = parse(Bool, cfg["normalize-advantages"]),
        clip_value_loss = parse(Bool, cfg["clip-value-loss"]),
        max_steps = parse(Int, cfg["max-steps"]),
        eval_runs = parse(Int, cfg["eval-runs"]),
        tree_queries = parse(Int, cfg["tree-queries"]),
        train_oos_iterations = parse(Int, cfg["train-oos-iterations"]),
        max_depth = parse(Int, cfg["max-depth"]),
        epsilon = parse(Float64, cfg["epsilon"]),
        transfer_weight = parse(Float64, cfg["transfer-weight"]),
        seed = parse(Int, cfg["seed"]),
        initial_state = cfg["initial-state"],
        output_dir = cfg["output-dir"],
        test = parse(Bool, cfg["test"]),
    )
    return parsed.test ? test_config(parsed) : parsed
end

split_nonempty(s::AbstractString) = String.(filter(!isempty, strip.(split(s, ","))))

function parse_players(spec::AbstractString)
    spec == "both" && return [1, 2]
    return map(split_nonempty(spec)) do item
        item in ("1", "p1", "P1") && return 1
        item in ("2", "p2", "P2") && return 2
        error("Unknown player spec $(item)")
    end
end

function test_config(cfg)
    return merge(cfg, (;
        targets = first(cfg.targets, min(length(cfg.targets), 1)),
        players = first(cfg.players, min(length(cfg.players), 1)),
        total_timesteps = 64,
        num_steps = 8,
        num_envs = 1,
        num_minibatches = 1,
        update_epochs = 1,
        eval_runs = 2,
        tree_queries = 2,
        max_depth = 2,
        max_steps = 4,
        output_dir = joinpath(cfg.output_dir, "smoke"),
    ))
end

function checkpoint_iteration(path::AbstractString)
    m = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(m) && error("Cannot parse checkpoint iteration from $(path)")
    return parse(Int, m.captures[1])
end

function checkpoint_paths()
    models_dir = joinpath(EXPERIMENT_DIR, "models_smoos")
    isdir(models_dir) || error("Missing model checkpoint directory: $(models_dir)")
    checkpoints = filter(p -> occursin(r"oracle\d+\.jld2$", basename(p)), readdir(models_dir; join=true))
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
    oracle_file = joinpath(EXPERIMENT_DIR, "oracle_smoos.jld2")
    isfile(oracle_file) || error("Missing oracle architecture file: $(oracle_file)")
    checkpoint = select_checkpoint(iter_spec)
    oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(oracle, checkpoint)
    return oracle, checkpoint_iteration(checkpoint), checkpoint
end

function transfer_tau(iter::Integer; oos_iterations::Integer, transfer_weight::Real)
    tau = 0.0
    for _ in 1:iter
        tau = AZ.advance_transfer_tau(tau, oos_iterations, transfer_weight)
    end
    return tau
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
    error("Unknown --initial-state $(spec); expected reference or game")
end

function fixed_policy(target::AbstractString, game, oracle, model_iter, cfg)
    if target == "network"
        return Tools.OracleStrategyPolicy(game, oracle), (; transfer_tau = 0.0)
    elseif target == "oos_value"
        search = AZ.SMOOSSearch(;
            oracle,
            oos_iterations = cfg.tree_queries,
            max_depth = cfg.max_depth,
            transfer_weight = 0.0,
            τ = 0.0,
            ϵ = _ -> cfg.epsilon,
        )
        return AZ.AlphaZeroPlanner(game, search), (; transfer_tau = 0.0)
    elseif target == "oos_zero"
        zero_oracle = Tools.ZeroSearchOracle(game)
        search = AZ.SMOOSSearch(;
            oracle = zero_oracle,
            oos_iterations = cfg.tree_queries,
            max_depth = cfg.max_depth,
            transfer_weight = 0.0,
            τ = 0.0,
            ϵ = _ -> cfg.epsilon,
        )
        return AZ.AlphaZeroPlanner(game, search), (; transfer_tau = 0.0)
    elseif target == "oos_transfer"
        tau = transfer_tau(
            model_iter;
            oos_iterations = cfg.train_oos_iterations,
            transfer_weight = cfg.transfer_weight,
        )
        search = AZ.SMOOSSearch(;
            oracle,
            oos_iterations = cfg.tree_queries,
            max_depth = cfg.max_depth,
            transfer_weight = cfg.transfer_weight,
            τ = tau,
            ϵ = _ -> cfg.epsilon,
        )
        return AZ.AlphaZeroPlanner(game, search), (; transfer_tau = tau)
    else
        error("Unknown target $(target)")
    end
end

function ppo_log_name(target::AbstractString, br_player::Int)
    fixed_player = MarkovGames.other_player(br_player)
    return "br_p$(br_player)__vs_$(target)_p$(fixed_player)"
end

function ppo_best_response_config(cfg, seed::Int, name::String)
    return Tools.PPOBestResponseConfig(;
        total_timesteps = cfg.total_timesteps,
        num_steps = cfg.num_steps,
        num_envs = cfg.num_envs,
        num_minibatches = cfg.num_minibatches,
        update_epochs = cfg.update_epochs,
        lr = cfg.lr,
        gamma = cfg.gamma,
        gae_lambda = cfg.gae_lambda,
        clip_coef = cfg.clip_coef,
        ent_coeff = cfg.ent_coeff,
        v_coef = cfg.v_coef,
        normalize_advantages = cfg.normalize_advantages,
        clip_value_loss = cfg.clip_value_loss,
        anneal_lr = cfg.anneal_lr,
        max_steps = cfg.max_steps,
        seed,
        name,
        log_dir = joinpath(cfg.output_dir, "tensorboard"),
    )
end

function rollout_eval(game, joint_policy, initialstates; runs::Int, max_steps::Int)
    return Tools.evaluate_joint_policy(
        game,
        joint_policy,
        runs;
        max_steps,
        initialstates,
        show_progress = false,
        proc_warn = false,
        parallel = false,
        accumulators = (
            StepCount(),
            DubinTools.DubinOutcome(),
        ),
        batch_accumulators = (
            MeanResult(:steps; name = :mean_steps),
            Tools.StdErrResult(:reward; name = :stderr_reward, init = zero(MarkovGames.reward_type(game))),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
    )
end

function evaluation_initialstates(rng, dist, n::Int)
    return [rand(rng, dist) for _ in 1:n]
end

function result_row(result)
    return Any[
        result.target,
        result.br_player,
        result.model_iter,
        result.checkpoint,
        result.cleanrl_rev,
        result.ppo_name,
        result.total_timesteps,
        result.num_envs,
        result.num_steps,
        result.max_steps,
        result.tree_queries,
        result.max_depth,
        result.transfer_weight,
        result.transfer_tau,
        result.br_reward,
        result.br_stderr_reward,
        result.p1_reward,
        result.p2_reward,
        result.attacker_goal_rate,
        result.tagged_rate,
        result.timeout_rate,
        result.mean_steps,
        result.model_path,
    ]
end

function run_one(target::String, br_player::Int, game, oracle, model_iter, checkpoint, cfg)
    fixed, target_meta = fixed_policy(target, game, oracle, model_iter, cfg)
    init_dist = initialstate_dist(game, cfg.initial_state)
    target_offset = sum(Int, codeunits(target))
    local_seed = cfg.seed + 10_000 * br_player + target_offset
    ppo_name = ppo_log_name(target, br_player)

    println("Training PPO BR: target=$(target) br_player=$(br_player) name=$(ppo_name)")
    br_result = Tools.train_ppo_best_response(
        game,
        fixed,
        br_player;
        initialstate_dist = init_dist,
        config = ppo_best_response_config(cfg, local_seed, ppo_name),
    )
    actor, critic = br_result.actor, br_result.critic

    eval_rng = Random.MersenneTwister(local_seed + 1)
    eval_states = evaluation_initialstates(eval_rng, init_dist, cfg.eval_runs)
    joint_policy = Tools.ppo_best_response_joint_policy(
        game,
        fixed,
        actor,
        br_player,
    )
    eval_result = rollout_eval(game, joint_policy, eval_states; runs = cfg.eval_runs, max_steps = cfg.max_steps)

    target_dir = joinpath(cfg.output_dir, target, "p$(br_player)")
    mkpath(target_dir)
    model_path = joinpath(target_dir, "ppo_br_actor_critic.jld2")
    metadata = Dict(
        "target" => target,
        "br_player" => br_player,
        "model_iter" => model_iter,
        "checkpoint" => checkpoint,
        "cleanrl_rev" => CLEANRL_REV,
        "ppo_name" => ppo_name,
        "total_timesteps" => cfg.total_timesteps,
        "num_envs" => cfg.num_envs,
        "num_steps" => cfg.num_steps,
        "max_steps" => cfg.max_steps,
        "tree_queries" => cfg.tree_queries,
        "max_depth" => cfg.max_depth,
        "transfer_weight" => cfg.transfer_weight,
        "transfer_tau" => target_meta.transfer_tau,
    )
    jldsave(model_path; actor, critic, metadata)

    br_reward = eval_result.reward[br_player]
    br_stderr = eval_result.stderr_reward[br_player]
    println(
        "eval target=$(target) p$(br_player) BR reward=$(round(br_reward; digits=4)) ",
        "stderr=$(round(br_stderr; digits=4)) steps=$(round(eval_result.mean_steps; digits=2)) ",
        "goal=$(round(eval_result.attacker_goal_rate; digits=3)) ",
        "tagged=$(round(eval_result.tagged_rate; digits=3))"
    )

    return (;
        target,
        br_player,
        model_iter,
        checkpoint,
        cleanrl_rev = CLEANRL_REV,
        ppo_name,
        total_timesteps = cfg.total_timesteps,
        num_envs = cfg.num_envs,
        num_steps = cfg.num_steps,
        max_steps = cfg.max_steps,
        tree_queries = cfg.tree_queries,
        max_depth = cfg.max_depth,
        transfer_weight = cfg.transfer_weight,
        transfer_tau = target_meta.transfer_tau,
        br_reward,
        br_stderr_reward = br_stderr,
        p1_reward = eval_result.reward[1],
        p2_reward = eval_result.reward[2],
        attacker_goal_rate = eval_result.attacker_goal_rate,
        tagged_rate = eval_result.tagged_rate,
        timeout_rate = eval_result.timeout_rate,
        mean_steps = eval_result.mean_steps,
        model_path,
    )
end

function write_summary(path, results)
    header = Any[
        "target",
        "br_player",
        "model_iter",
        "checkpoint",
        "cleanrl_rev",
        "ppo_name",
        "total_timesteps",
        "num_envs",
        "num_steps",
        "max_steps",
        "tree_queries",
        "max_depth",
        "transfer_weight",
        "transfer_tau",
        "br_reward",
        "br_stderr_reward",
        "p1_reward",
        "p2_reward",
        "attacker_goal_rate",
        "tagged_rate",
        "timeout_rate",
        "mean_steps",
        "model_path",
    ]
    rows = Any[header]
    append!(rows, result_row.(results))
    mkpath(dirname(path))
    writedlm(path, rows, ',')
    return path
end

function main(args)
    cfg = parse_args(args)
    game = DubinMG(V = (1.0, 1.0))
    oracle, model_iter, checkpoint = load_checkpoint_oracle(cfg.iter)
    mkpath(cfg.output_dir)
    println("Loaded checkpoint: $(checkpoint)")
    println("CleanRL fork expected at: https://github.com/WhiffleFish/CleanRL.jl#$(CLEANRL_REV)")

    results = NamedTuple[]
    for target in cfg.targets
        for br_player in cfg.players
            push!(results, run_one(target, br_player, game, oracle, model_iter, checkpoint, cfg))
        end
    end

    summary_path = write_summary(joinpath(cfg.output_dir, "summary.csv"), results)
    println("Wrote summary: $(summary_path)")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
