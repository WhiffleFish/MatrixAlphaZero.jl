using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

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
const EXPERIMENT_NAME = "dubin-2026-07-16"
const EXPERIMENT_GROUP = "$(EXPERIMENT_NAME)-rm-plus-transfer-500"
const SEED = 0

@isdefined(CONDITION) || error("Define CONDITION before including experiment.jl")
CONDITION == :transfer_default || error("Unsupported CONDITION=$(CONDITION)")

const SEARCH_NAME = "rm_plus_transfer"
const LOSS_SCALED_TRANSFER = AZ.LossScaledTransfer()

args = ExperimentTools.parse_commandline(
    max_steps=10_000_000,
    num_steps=1024 * 8,
    update_epochs=2,
    num_batches=4,
    tree_queries=500,
    max_depth=5,
    sim_depth=50,
    runs=100,
    every=10,
)

p = addprocs(args["addprocs"])

max_steps = args["max_steps"]
num_steps = args["num_steps"]
update_epochs = args["update_epochs"]
num_batches = args["num_batches"]
tree_queries = args["tree_queries"]
search_depth = args["max_depth"]
sim_depth = args["sim_depth"]
eval_runs = args["runs"]
eval_every = args["every"]

width = 32
lr = 3f-4
lr_decay = 0.999f0
lr_min = 1f-5
lr_max = lr
ema = false
ema_decay = 0.98f0
gae_lambda = 0.95
epsilon_decay = 1 - 1e-3
epsilon_schedule = t -> max(0.3 * epsilon_decay^(t - 1), 0.1)
max_time = Inf
backup = :sample
value_target = :search
value_weight = 1.0f0
regret_weight = 0.1f0
strategy_weight = 0.5f0
transfer_payoff_bound = 2.0
critic_type = "scalar"

@everywhere begin
    using Distributed
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using POSGModels.Dubin
    using POSGModels.StaticArrays
    using Random
end
@everywhere Random.seed!($SEED + Distributed.myid())

game = DubinMG(V=(1.0, 1.0))
na1, na2 = length.(actions(game))
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

function state_network(width, output_dim)
    return Chain(
        Dense(8 => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
        Dense(width => output_dim),
    )
end

function init_oracle(width, na1, na2; value_weight, regret_weight, strategy_weight)
    critic = state_network(width, 1)
    regret_head = AZ.MultiActor(
        state_network(width, na1),
        state_network(width, na2),
    )
    strategy_head = AZ.MultiActor(
        state_network(width, na1),
        state_network(width, na2),
    )
    return AZ.FittedRegretModel(
        regret_head,
        strategy_head,
        critic;
        value_weight,
        regret_weight,
        strategy_weight,
    )
end

function callback_transfer_value(info, name::Symbol)
    isnothing(info) && return 0.0
    return hasproperty(info, name) ? Float64(getproperty(info, name)) : 0.0
end

function build_search(oracle; epsilon, info=nothing)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries,
        max_depth=search_depth,
        max_time,
        search_style=AZ.RegretMatchingSearch(; backup, method=AZ.Plus()),
        value_target,
        ϵ=epsilon,
        τ=callback_transfer_value(info, :transfer_source_mass),
        transfer_weight=0.0,
        transfer_payoff_bound,
        loss_scaled_transfer=LOSS_SCALED_TRANSFER,
        regret_confidence=callback_transfer_value(info, :transfer_regret_confidence),
        strategy_confidence=callback_transfer_value(info, :transfer_strategy_confidence),
    )
end

function add_scalar_metric!(pairs, prefix::AbstractString, name::Symbol, value)
    value isa Number || return pairs
    isfinite(value) || return pairs
    push!(pairs, Symbol(prefix, "/", name) => value)
    return pairs
end

function prefixed_az_eval_metrics(result::NamedTuple, prefix::AbstractString, az_player::Int)
    pairs = Pair{Symbol,Any}[]
    add_scalar_metric!(pairs, prefix, :reward, result.reward[az_player])
    add_scalar_metric!(pairs, prefix, :mean_steps, result.mean_steps)
    add_scalar_metric!(pairs, prefix, :attacker_goal_rate, result.attacker_goal_rate)
    add_scalar_metric!(pairs, prefix, :tagged_rate, result.tagged_rate)
    add_scalar_metric!(pairs, prefix, :timeout_rate, result.timeout_rate)
    extra_prefix = replace(prefix, "eval/" => "eval_extra/"; count=1)
    add_scalar_metric!(pairs, extra_prefix, :stderr_reward, result.stderr_reward[az_player])
    return (; pairs...)
end

function print_eval_summary(iter, az_p1_result, az_p2_result)
    println(
        "eval iter $(iter): ",
        "AZ p1 reward=$(round(az_p1_result.reward[1]; digits=3)) ",
        "goal=$(round(az_p1_result.attacker_goal_rate; digits=3)) ",
        "tagged=$(round(az_p1_result.tagged_rate; digits=3)) ",
        "steps=$(round(az_p1_result.mean_steps; digits=1)); ",
        "AZ p2 reward=$(round(az_p2_result.reward[2]; digits=3)) ",
        "goal=$(round(az_p2_result.attacker_goal_rate; digits=3)) ",
        "tagged=$(round(az_p2_result.tagged_rate; digits=3)) ",
        "steps=$(round(az_p2_result.mean_steps; digits=1))",
    )
    return nothing
end

function eval_search_epsilon(info::NamedTuple)
    hasproperty(info, :exploration_epsilon) && return info.exploration_epsilon
    hasproperty(info, :update) && return epsilon_schedule(max(info.update, 1))
    return epsilon_schedule(1)
end

struct StatRolloutEvalCallback{G,S,W}
    game::G
    s0::S
    n::Int
    max_steps::Int
    eval_every::Int
    wandb_cb::W
end

function rollout_eval(cb::StatRolloutEvalCallback, joint_policy)
    return Tools.evaluate_joint_policy(
        cb.game,
        joint_policy,
        cb.n;
        max_steps=cb.max_steps,
        initialstates=fill(cb.s0, cb.n),
        show_progress=false,
        proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), DubinTools.DubinOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(
                :reward;
                name=:stderr_reward,
                init=zero(MarkovGames.reward_type(cb.game)),
            ),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
    )
end

function (cb::StatRolloutEvalCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    iszero(mod(info.iter, cb.eval_every)) || return
    search = build_search(info.oracle; epsilon=_ -> eval_search_epsilon(info), info)
    planner = AZ.AlphaZeroPlanner(cb.game, search)
    az_p1 = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(planner, 1),
        DubinTools.dubin_defender_heuristic(cb.game),
    )
    az_p2 = Tools.JointPolicy(
        DubinTools.dubin_attacker_heuristic(cb.game),
        Tools.SinglePlayerAlphaZeroPolicy(planner, 2),
    )
    Random.seed!(SEED + 10_000_000 + 2 * info.iter)
    az_p1_result = rollout_eval(cb, az_p1)
    Random.seed!(SEED + 10_000_001 + 2 * info.iter)
    az_p2_result = rollout_eval(cb, az_p2)
    metrics = merge(
        prefixed_az_eval_metrics(az_p1_result, "eval/az_p1_vs_heuristic", 1),
        prefixed_az_eval_metrics(az_p2_result, "eval/heuristic_vs_az_p2", 2),
    )
    print_eval_summary(info.iter, az_p1_result, az_p2_result)
    isnothing(cb.wandb_cb) || cb.wandb_cb(merge((; iter=info.iter), metrics))
    return nothing
end

Random.seed!(SEED)
oracle = init_oracle(
    width,
    na1,
    na2;
    value_weight,
    regret_weight,
    strategy_weight,
)
Random.seed!(SEED + 1_000_000)
models_dir = joinpath(@__DIR__, "models_$(SEARCH_NAME)")
mkpath(models_dir)
jldsave(joinpath(@__DIR__, "oracle_$(SEARCH_NAME).jld2"); oracle)

search = build_search(oracle; epsilon=epsilon_schedule)
solver = AZ.AlphaZeroSolver(
    search=search,
    max_steps=max_steps,
    num_steps=num_steps,
    sim_depth=sim_depth,
    update_epochs=update_epochs,
    num_batches=num_batches,
    lr=lr,
    lr_decay=lr_decay,
    lr_min=lr_min,
    lr_max=lr_max,
    ema=ema,
    ema_decay=ema_decay,
    gae_lambda=gae_lambda,
    rng=Random.MersenneTwister(SEED),
)

wandb_config = Dict{String,Any}(
    "experiment" => EXPERIMENT_NAME,
    "condition" => string(CONDITION),
    "seed" => SEED,
    "search/name" => SEARCH_NAME,
    "search/type" => "MCTSSearch",
    "search/tree_queries" => search.tree_queries,
    "search/max_depth" => search.max_depth,
    "search/max_time" => search.max_time,
    "search/backup" => string(search.search_style.backup),
    "search/method" => "plus",
    "search/value_target" => string(search.value_target),
    "game" => "DubinMG",
    "sim_depth" => sim_depth,
    "max_steps" => max_steps,
    "num_steps" => num_steps,
    "update_epochs" => update_epochs,
    "num_batches" => num_batches,
    "width" => width,
    "oracle/shared_trunk" => false,
    "oracle/state_network" => "8-$(width)-$(width)-$(width)-output",
    "lr" => lr,
    "lr_decay" => lr_decay,
    "lr_min" => lr_min,
    "lr_max" => lr_max,
    "ema" => ema,
    "ema_decay" => ema_decay,
    "gae_lambda" => gae_lambda,
    "oracle/value_weight" => oracle.value_weight,
    "critic_type" => critic_type,
    "epsilon_initial" => epsilon_schedule(1),
    "epsilon_decay" => epsilon_decay,
    "eval_runs" => eval_runs,
    "eval_every" => eval_every,
    "oracle/regret_weight" => oracle.regret_weight,
    "oracle/strategy_weight" => oracle.strategy_weight,
    "search/transfer_payoff_bound" => search.transfer_payoff_bound,
    "search/loss_scaled_transfer/regret_scale" => LOSS_SCALED_TRANSFER.regret_scale,
    "search/loss_scaled_transfer/strategy_scale" => LOSS_SCALED_TRANSFER.strategy_scale,
    "search/loss_scaled_transfer/reach_power" => LOSS_SCALED_TRANSFER.reach_power,
    "search/loss_scaled_transfer/confidence_ema_decay" => LOSS_SCALED_TRANSFER.confidence_ema_decay,
    "search/loss_scaled_transfer/loss_tail_fraction" => LOSS_SCALED_TRANSFER.loss_tail_fraction,
)

wandb_cb = if get(ENV, "WANDB_API_KEY", "") != ""
    WandbCallback(
        project="Matrix AlphaZero",
        group=EXPERIMENT_GROUP,
        config=wandb_config,
    )
else
    @warn "WANDB_API_KEY not set - skipping W&B logging"
    nothing
end

stat_eval_cb = StatRolloutEvalCallback(
    game,
    s0,
    eval_runs,
    sim_depth,
    eval_every,
    wandb_cb,
)
callbacks = isnothing(wandb_cb) ?
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), stat_eval_cb) :
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), wandb_cb, stat_eval_cb)

solve(solver, game; s0=Deterministic(s0), cb=callbacks)
isnothing(wandb_cb) || close(wandb_cb)
rmprocs(p)
