using Pkg
Pkg.activate("experiments")

using DelimitedFiles
using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using ProgressMeter
using Random

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_DIR = @__DIR__

function parse_args(args)
    cfg = Dict{String,String}(
        "runs" => "100",
        "weights" => "0:0.1:1",
        "iter" => "latest",
        "tree-queries" => "1000",
        "train-oos-iterations" => "1000",
        "max-depth" => "5",
        "max-steps" => "50",
        "epsilon" => "0.0",
        "seed" => "20260621",
        "show-progress" => "true",
        "output" => joinpath(EXPERIMENT_DIR, "transfer_weight_sweep.csv"),
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
    return (;
        runs = parse(Int, cfg["runs"]),
        weights = parse_weights(cfg["weights"]),
        iter = cfg["iter"],
        tree_queries = parse(Int, cfg["tree-queries"]),
        train_oos_iterations = parse(Int, cfg["train-oos-iterations"]),
        max_depth = parse(Int, cfg["max-depth"]),
        max_steps = parse(Int, cfg["max-steps"]),
        epsilon = parse(Float64, cfg["epsilon"]),
        seed = parse(Int, cfg["seed"]),
        show_progress = parse(Bool, cfg["show-progress"]),
        output = cfg["output"],
    )
end

function parse_weights(spec::AbstractString)
    parts = split(spec, ":")
    if length(parts) == 3
        lo, step, hi = parse.(Float64, parts)
        step > 0 || error("Weight step must be positive")
        hi >= lo || error("Weight range must be increasing")
        n = floor(Int, (hi - lo) / step)
        weights = [lo + step * i for i in 0:n]
        if isempty(weights) || !isapprox(last(weights), hi; atol=1e-12, rtol=0.0)
            push!(weights, hi)
        end
        return weights
    end
    return parse.(Float64, split(spec, ","))
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
    if iter_spec == "latest"
        return last(checkpoints)
    end
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

function rollout_eval(game, joint_policy, s0; runs::Int, max_steps::Int)
    return Tools.evaluate_joint_policy(
        game,
        joint_policy,
        runs;
        max_steps,
        initialstates = fill(s0, runs),
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

function evaluate_weight(
        game,
        s0,
        oracle;
        weight::Float64,
        model_iter::Int,
        runs::Int,
        tree_queries::Int,
        train_oos_iterations::Int,
        max_depth::Int,
        max_steps::Int,
        epsilon::Float64,
        seed::Int,
    )
    tau = transfer_tau(model_iter; oos_iterations=train_oos_iterations, transfer_weight=weight)
    search = AZ.SMOOSSearch(;
        oracle,
        oos_iterations = tree_queries,
        max_depth,
        transfer_weight = weight,
        τ = tau,
        ϵ = _ -> epsilon,
    )
    planner = AZ.AlphaZeroPlanner(game, search)

    az_p1 = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(planner, 1),
        DubinTools.dubin_defender_heuristic(game),
    )
    az_p2 = Tools.JointPolicy(
        DubinTools.dubin_attacker_heuristic(game),
        Tools.SinglePlayerAlphaZeroPolicy(planner, 2),
    )

    Random.seed!(seed)
    az_p1_result = rollout_eval(game, az_p1, s0; runs, max_steps)
    Random.seed!(seed)
    az_p2_result = rollout_eval(game, az_p2, s0; runs, max_steps)

    return (;
        weight,
        tau,
        az_p1 = az_p1_result,
        az_p2 = az_p2_result,
    )
end

function result_row(result, cfg, model_iter)
    az_p1 = result.az_p1
    az_p2 = result.az_p2
    return Any[
        result.weight,
        result.tau,
        model_iter,
        cfg.runs,
        cfg.tree_queries,
        cfg.train_oos_iterations,
        cfg.max_depth,
        cfg.max_steps,
        cfg.epsilon,
        az_p1.reward[1],
        az_p1.stderr_reward[1],
        az_p1.attacker_goal_rate,
        az_p1.tagged_rate,
        az_p1.timeout_rate,
        az_p1.mean_steps,
        az_p2.reward[2],
        az_p2.stderr_reward[2],
        az_p2.attacker_goal_rate,
        az_p2.tagged_rate,
        az_p2.timeout_rate,
        az_p2.mean_steps,
    ]
end

function main(args)
    cfg = parse_args(args)
    game = DubinMG(V = (1.0, 1.0))
    s0 = initial_dubin_state()
    oracle, model_iter, checkpoint = load_checkpoint_oracle(cfg.iter)

    header = [
        "transfer_weight",
        "transfer_tau",
        "model_iter",
        "runs",
        "tree_queries",
        "train_oos_iterations",
        "max_depth",
        "max_steps",
        "epsilon",
        "az_p1_vs_heuristic_reward",
        "az_p1_vs_heuristic_stderr_reward",
        "az_p1_vs_heuristic_attacker_goal_rate",
        "az_p1_vs_heuristic_tagged_rate",
        "az_p1_vs_heuristic_timeout_rate",
        "az_p1_vs_heuristic_mean_steps",
        "heuristic_vs_az_p2_reward",
        "heuristic_vs_az_p2_stderr_reward",
        "heuristic_vs_az_p2_attacker_goal_rate",
        "heuristic_vs_az_p2_tagged_rate",
        "heuristic_vs_az_p2_timeout_rate",
        "heuristic_vs_az_p2_mean_steps",
    ]

    rows = Any[header]
    println("Loaded checkpoint: $(checkpoint)")
    progress = Progress(
        length(cfg.weights);
        desc = "Transfer sweep: ",
        showspeed = true,
        enabled = cfg.show_progress,
    )
    for weight in cfg.weights
        result = evaluate_weight(
            game,
            s0,
            oracle;
            weight,
            model_iter,
            runs = cfg.runs,
            tree_queries = cfg.tree_queries,
            train_oos_iterations = cfg.train_oos_iterations,
            max_depth = cfg.max_depth,
            max_steps = cfg.max_steps,
            epsilon = cfg.epsilon,
            seed = cfg.seed,
        )
        push!(rows, result_row(result, cfg, model_iter))
        next!(progress; showvalues = [
            (:weight, round(weight; digits=3)),
            (:tau, round(result.tau; digits=3)),
            (:az_p1_reward, round(result.az_p1.reward[1]; digits=3)),
            (:az_p2_reward, round(result.az_p2.reward[2]; digits=3)),
        ])
    end
    finish!(progress)

    mkpath(dirname(cfg.output))
    writedlm(cfg.output, rows, ',')
    println("Wrote: $(cfg.output)")
    return nothing
end

main(ARGS)
