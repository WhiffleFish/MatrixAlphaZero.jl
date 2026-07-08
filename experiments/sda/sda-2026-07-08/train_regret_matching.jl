using Pkg
Pkg.activate("experiments")

using Distributed
using Distributions
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics
using Random

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const EXPERIMENT_NAME = "sda-2026-07-08"
const SEARCH_NAME = "regret_matching"

args = ExperimentTools.parse_commandline(
    max_steps = 10_000_000,
    num_steps = 1024 * 8,
    update_epochs = 2,
    num_batches = 4,
    tree_queries = 500,
    max_depth = 5,
    sim_depth = 50,
    runs = 50,
    every = 5,
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
epsilon_schedule = t -> max(0.3 * (epsilon_decay ^ (t - 1)), 0.1)
max_time = Inf
backup = :sample
value_target = :search
value_weight = 1.0f0
critic_type = "scalar"

@everywhere begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using SDAGames.SNRGame
    using SDAGames.SatelliteDynamics
end

# Random LEO/MEO orbits in the equatorial plane for both satellites, matching the
# established SNR-SDA setup (experiments/sda/sda-2026-02-20/train.jl).
d_observer = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π,
    ])
end

d_target = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π,
    ])
end

game = SNRSDAGame(observer = d_observer, target = d_target, altitude_bounds = (100e3, 2e7))

# Value-only oracle: RegretMatchingSearch builds its action distributions from
# scratch via regret matching over q = r + γV̂, so no actor/policy head is needed.
# convert_s returns a 15-dim feature vector (observer 6D, target 6D, sun 3D).
function init_oracle(width; value_weight)
    trunk = Chain(Dense(15 => width, tanh), Dense(width => width, tanh))
    critic = Chain(Dense(width => width, tanh), Dense(width => 1))
    return AZ.CriticOnly(trunk, critic; value_weight)
end

function add_scalar_metric!(pairs, prefix::AbstractString, name::Symbol, value)
    value isa Number || return pairs
    isfinite(value) || return pairs
    push!(pairs, Symbol(prefix, "/", name) => value)
    return pairs
end

function prefixed_az_eval_metrics(result::NamedTuple, prefix::AbstractString, az_player::Int)
    pairs = Pair{Symbol,Any}[]
    # The SNR-SDA game returns a scalar (zero-sum) reward = the observer's (player 1)
    # payoff, so the target's reward is its negation.
    observer_reward = result.reward[1]
    az_reward = isone(az_player) ? observer_reward : -observer_reward

    add_scalar_metric!(pairs, prefix, :reward, az_reward)
    add_scalar_metric!(pairs, prefix, :mean_steps, result.mean_steps)
    add_scalar_metric!(pairs, prefix, :detection_rate, result.detected_rate)
    add_scalar_metric!(pairs, prefix, :target_escaped_rate, result.target_escaped_rate)
    add_scalar_metric!(pairs, prefix, :observer_lost_rate, result.observer_lost_rate)

    extra_prefix = replace(prefix, "eval/" => "eval_extra/"; count = 1)
    add_scalar_metric!(pairs, extra_prefix, :stderr_reward, result.stderr_reward[1])
    return (; pairs...)
end

function print_eval_summary(iter, az_obs_result, az_tar_result)
    az_obs_reward = az_obs_result.reward[1]
    az_tar_reward = -az_tar_result.reward[1]
    println(
        "eval iter $(iter): ",
        "AZ observer reward=$(round(az_obs_reward; digits=3)) ",
        "detect=$(round(az_obs_result.detected_rate; digits=3)) ",
        "steps=$(round(az_obs_result.mean_steps; digits=1)); ",
        "AZ target reward=$(round(az_tar_reward; digits=3)) ",
        "detect=$(round(az_tar_result.detected_rate; digits=3)) ",
        "escaped=$(round(az_tar_result.target_escaped_rate; digits=3)) ",
        "steps=$(round(az_tar_result.mean_steps; digits=1))",
    )
    return nothing
end

function eval_search_epsilon(info::NamedTuple)
    if hasproperty(info, :exploration_epsilon)
        return info.exploration_epsilon
    elseif hasproperty(info, :update)
        return epsilon_schedule(max(info.update, 1))
    else
        return epsilon_schedule(1)
    end
end

struct StatRolloutEvalCallback{G,S,W}
    game::G
    initialstates::S
    n::Int
    max_steps::Int
    eval_every::Int
    tree_queries::Int
    search_depth::Int
    wandb_cb::W
end

function rollout_eval(cb::StatRolloutEvalCallback, joint_policy)
    return Tools.evaluate_joint_policy(
        cb.game,
        joint_policy,
        cb.n;
        max_steps = cb.max_steps,
        initialstates = cb.initialstates,
        show_progress = false,
        proc_warn = false,
        parallel = false,
        accumulators = (
            StepCount(),
            SDAOutcome(),
        ),
        batch_accumulators = (
            MeanResult(:steps; name = :mean_steps),
            Tools.StdErrResult(:reward; name = :stderr_reward, init = zero(MarkovGames.reward_type(cb.game))),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
    )
end

function (cb::StatRolloutEvalCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    iszero(mod(info.iter, cb.eval_every)) || return
    search = AZ.MCTSSearch(;
        oracle = info.oracle,
        tree_queries = cb.tree_queries,
        max_depth = cb.search_depth,
        max_time,
        ϵ = _ -> eval_search_epsilon(info),
        search_style = AZ.RegretMatchingSearch(; backup),
        value_target,
    )
    planner = AZ.AlphaZeroPlanner(cb.game, search)

    az_observer = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(planner, 1),
        Tools.sda_no_burn_heuristic(cb.game, 2),
    )
    az_target = Tools.JointPolicy(
        Tools.sda_no_burn_heuristic(cb.game, 1),
        Tools.SinglePlayerAlphaZeroPolicy(planner, 2),
    )

    az_obs_result = rollout_eval(cb, az_observer)
    az_tar_result = rollout_eval(cb, az_target)
    metrics = merge(
        prefixed_az_eval_metrics(az_obs_result, "eval/az_observer_vs_heuristic", 1),
        prefixed_az_eval_metrics(az_tar_result, "eval/heuristic_vs_az_target", 2),
    )
    print_eval_summary(info.iter, az_obs_result, az_tar_result)
    isnothing(cb.wandb_cb) || cb.wandb_cb(merge((; iter = info.iter), metrics))
    return nothing
end

Random.seed!(0)
oracle = init_oracle(
    width;
    value_weight,
)
experiment_dir = @__DIR__
models_dir = joinpath(experiment_dir, "models_$(SEARCH_NAME)")
mkpath(experiment_dir)
jldsave(joinpath(experiment_dir, "oracle_$(SEARCH_NAME).jld2"); oracle)

# Fixed set of evaluation initial states sampled from the game distribution, so the
# eval metric is measured against the same orbital configurations at every iteration.
eval_rng = MersenneTwister(1)
eval_initialstates = [rand(eval_rng, initialstate(game)) for _ in 1:eval_runs]

search = AZ.MCTSSearch(;
    oracle,
    tree_queries,
    max_depth = search_depth,
    max_time,
    ϵ = epsilon_schedule,
    search_style = AZ.RegretMatchingSearch(; backup),
    value_target,
)

solver = AZ.AlphaZeroSolver(
    search = search,
    max_steps = max_steps,
    num_steps = num_steps,
    sim_depth = sim_depth,
    update_epochs = update_epochs,
    num_batches = num_batches,
    lr = lr,
    lr_decay = lr_decay,
    lr_min = lr_min,
    lr_max = lr_max,
    ema = ema,
    ema_decay = ema_decay,
    gae_lambda = gae_lambda,
)

wandb_cb = if get(ENV, "WANDB_API_KEY", "") != ""
    WandbCallback(
        project = "Matrix AlphaZero",
        group = "$(EXPERIMENT_NAME)-$(SEARCH_NAME)",
        config = Dict(
            "experiment" => EXPERIMENT_NAME,
            "search/name" => SEARCH_NAME,
            "search/type" => "MCTSSearch",
            "game" => "SNRSDAGame",
            "search/tree_queries" => search.tree_queries,
            "search/max_depth" => search.max_depth,
            "search/max_time" => search.max_time,
            "search/backup" => string(search.search_style.backup),
            "search/value_target" => string(search.value_target),
            "sim_depth" => sim_depth,
            "max_steps" => max_steps,
            "num_steps" => num_steps,
            "update_epochs" => update_epochs,
            "num_batches" => num_batches,
            "width" => width,
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
        ),
    )
else
    @warn "WANDB_API_KEY not set - skipping W&B logging"
    nothing
end

stat_eval_cb = StatRolloutEvalCallback(
    game,
    eval_initialstates,
    eval_runs,
    sim_depth,
    eval_every,
    tree_queries,
    search_depth,
    wandb_cb,
)

cb = if isnothing(wandb_cb)
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), stat_eval_cb)
else
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), wandb_cb, stat_eval_cb)
end

solve(solver, game; s0 = initialstate(game), cb)
isnothing(wandb_cb) || close(wandb_cb)

rmprocs(p)
