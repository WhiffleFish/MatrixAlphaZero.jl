using Pkg
Pkg.activate("experiments")

using Distributed
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_NAME = "dubin-2026-06-01"

args = ExperimentTools.parse_commandline(
    max_steps = 10_000_000,
    num_steps = 1024 * 8,
    update_epochs = 1,
    num_batches = 4,
    tree_queries = 500,
    max_depth = 50,
    runs = 16,
    every = 1,
)

p = addprocs(args["addprocs"])

max_steps = args["max_steps"]
num_steps = args["num_steps"]
update_epochs = args["update_epochs"]
num_batches = args["num_batches"]
oos_iterations = args["tree_queries"]
max_depth = args["max_depth"]
eval_runs = args["runs"]
eval_every = args["every"]

width = 32
lr = 3f-4
lr_decay = 0.98f0
ema_decay = 0.98f0
gae_lambda = 0.95
epsilon_schedule = t -> 0.3 * (0.90 ^ (t - 1))

@everywhere begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using POSGModels.Dubin
    using POSGModels.StaticArrays
end

game = DubinMG(V = (1.0, 1.0))
na1, na2 = length.(actions(game))
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

function init_oracle(width, na1, na2)
    trunk = Chain(Dense(8 => width, tanh), Dense(width => width, tanh))
    regret_head = AZ.MultiActor(
        Chain(Dense(width => width, tanh), Dense(width => na1)),
        Chain(Dense(width => width, tanh), Dense(width => na2)),
    )
    strategy_head = AZ.MultiActor(
        Chain(Dense(width => width, tanh), Dense(width => na1)),
        Chain(Dense(width => width, tanh), Dense(width => na2)),
    )
    critic = AZ.HLGaussCritic(
        Chain(Dense(width => width, tanh), Dense(width => 16)),
        -1,
        1,
        16,
    )
    return AZ.FittedRegretModel(trunk, regret_head, strategy_head, critic)
end

function prefix_metrics(result::NamedTuple, prefix::AbstractString)
    pairs = Pair{Symbol,Any}[]
    for k in propertynames(result)
        v = getproperty(result, k)
        if v isa AbstractVector{<:Number}
            for i in eachindex(v)
                push!(pairs, Symbol(prefix, "/", k, "_p", i) => v[i])
            end
        else
            push!(pairs, Symbol(prefix, "/", k) => v)
        end
    end
    return (; pairs...)
end

struct StatRolloutEvalCallback{G,S,W}
    game::G
    s0::S
    n::Int
    max_steps::Int
    eval_every::Int
    oos_iterations::Int
    search_depth::Int
    wandb_cb::W
end

function rollout_eval(cb::StatRolloutEvalCallback, joint_policy)
    return Tools.evaluate_joint_policy(
        cb.game,
        joint_policy,
        cb.n;
        max_steps = cb.max_steps,
        initialstates = fill(cb.s0, cb.n),
        show_progress = false,
        proc_warn = false,
        parallel = false,
        accumulators = (
            StepCount(),
            DubinTools.DubinOutcome(),
        ),
        batch_accumulators = (
            MeanResult(:steps; name = :mean_steps),
            Tools.StdErrResult(:reward; name = :stderr_reward, init = zero(MarkovGames.reward_type(cb.game))),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
    )
end

function (cb::StatRolloutEvalCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    iszero(info.iter % cb.eval_every) || return
    planner = AZ.AlphaZeroPlanner(
        cb.game,
        info.oracle;
        oos_iterations = cb.oos_iterations,
        max_depth = cb.search_depth,
    )

    az_p1 = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(planner, 1),
        DubinTools.dubin_defender_heuristic(cb.game),
    )
    az_p2 = Tools.JointPolicy(
        DubinTools.dubin_attacker_heuristic(cb.game),
        Tools.SinglePlayerAlphaZeroPolicy(planner, 2),
    )

    metrics = merge(
        prefix_metrics(rollout_eval(cb, az_p1), "stat_rollout/az_p1_vs_heuristic"),
        prefix_metrics(rollout_eval(cb, az_p2), "stat_rollout/heuristic_vs_az_p2"),
    )
    println("stat rollout eval iter $(info.iter): ", metrics)
    isnothing(cb.wandb_cb) || cb.wandb_cb(merge((; iter = info.iter), metrics))
    return nothing
end

Random.seed!(0)
oracle = init_oracle(width, na1, na2)
experiment_dir = @__DIR__
models_dir = joinpath(experiment_dir, "models")
mkpath(experiment_dir)
jldsave(joinpath(experiment_dir, "oracle.jld2"); oracle)

solver = AZ.AlphaZeroSolver(
    oracle = oracle,
    max_steps = max_steps,
    num_steps = num_steps,
    update_epochs = update_epochs,
    num_batches = num_batches,
    lr = lr,
    lr_decay = lr_decay,
    ema_decay = ema_decay,
    gae_lambda = gae_lambda,
    smoos_params = AZ.SMOOSParams(;
        oracle,
        oos_iterations,
        max_depth,
        ϵ = epsilon_schedule,
    ),
)

wandb_cb = if get(ENV, "WANDB_API_KEY", "") != ""
    WandbCallback(
        project = "Matrix AlphaZero",
        name = "$(EXPERIMENT_NAME)-regret-transfer-smoos",
        group = EXPERIMENT_NAME,
        config = Dict(
            "experiment" => EXPERIMENT_NAME,
            "game" => "DubinMG",
            "oos_iterations" => oos_iterations,
            "max_depth" => max_depth,
            "max_steps" => max_steps,
            "num_steps" => num_steps,
            "update_epochs" => update_epochs,
            "num_batches" => num_batches,
            "width" => width,
            "lr" => lr,
            "lr_decay" => lr_decay,
            "ema_decay" => ema_decay,
            "gae_lambda" => gae_lambda,
            "epsilon_initial" => epsilon_schedule(1),
            "epsilon_decay" => 0.90,
            "stat_rollout_runs" => eval_runs,
            "stat_rollout_every" => eval_every,
        ),
    )
else
    @warn "WANDB_API_KEY not set - skipping W&B logging"
    nothing
end

stat_eval_cb = StatRolloutEvalCallback(
    game,
    s0,
    eval_runs,
    max_depth,
    eval_every,
    oos_iterations,
    max_depth,
    wandb_cb,
)

cb = if isnothing(wandb_cb)
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), stat_eval_cb)
else
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), wandb_cb, stat_eval_cb)
end

solve(solver, game; s0 = Deterministic(s0), cb)
isnothing(wandb_cb) || close(wandb_cb)

rmprocs(p)
