using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributed
using Distributions
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics
using Statistics

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const EXPERIMENT_NAME = "sda-2026-07-20"
const EXPERIMENT_GROUP = "$(EXPERIMENT_NAME)-rm-plus-no-transfer-train-100q-prior100"
const SEARCH_NAME = "rm_plus_no_transfer_train"
const SEED = 0
const SDAGAMES_TREE_SHA = "3b41ec411b327b4889cca64648a38d1b1aa7e47f"
const EVAL_EPSILON = 0.0

@isdefined(CONDITION) || error("Define CONDITION before including experiment.jl")
CONDITION == :no_transfer_train || error("Unsupported CONDITION=$(CONDITION)")

args = ExperimentTools.parse_commandline(
    max_steps=10_000_000,
    num_steps=1024 * 8,
    update_epochs=2,
    num_batches=4,
    tree_queries=500,
    inference_tree_queries=100,
    max_depth=5,
    sim_depth=50,
    runs=100,
    every=10,
    prior_scale=100.0,
)

p = addprocs(args["addprocs"])

max_steps = args["max_steps"]
num_steps = args["num_steps"]
update_epochs = args["update_epochs"]
num_batches = args["num_batches"]
tree_queries = args["tree_queries"]
inference_tree_queries = args["inference_tree_queries"]
search_depth = args["max_depth"]
sim_depth = args["sim_depth"]
eval_runs = args["runs"]
eval_every = args["every"]
inference_prior_scale = Float64(args["prior_scale"])
tree_queries > 0 || throw(ArgumentError("tree_queries must be positive"))
inference_tree_queries > 0 || throw(ArgumentError("inference_tree_queries must be positive"))
0 <= inference_prior_scale <= tree_queries || throw(ArgumentError(
    "prior_scale must be in [0, tree_queries] so the equivalent transfer weight is at most 1",
))

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
    using Random
end
@everywhere Random.seed!($SEED + Distributed.myid())

# Keep the July-10 2-D SDA task exactly: circular equatorial initial orbits with
# independently sampled altitude and phase for observer and target.
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

game = SNRGameSimple(
    observer=d_observer,
    target=d_target,
    altitude_bounds=(100e3, 2e7),
)
na1, na2 = length.(actions(game))

function state_network(width, output_dim)
    return Chain(
        Dense(16 => width, tanh),
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

function build_search(oracle; epsilon, prior_scale=0.0, queries=tree_queries)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth=search_depth,
        max_time,
        search_style=AZ.RegretMatchingSearch(; backup, method=AZ.Plus()),
        value_target,
        ϵ=epsilon,
        prior_scale,
    )
end

function add_scalar_metric!(pairs, prefix::AbstractString, name::Symbol, value)
    value isa Number || return pairs
    isfinite(value) || return pairs
    push!(pairs, Symbol(prefix, "/", name) => value)
    return pairs
end

function prefixed_eval_metrics(result::NamedTuple, prefix::AbstractString, player::Int)
    pairs = Pair{Symbol,Any}[]
    add_scalar_metric!(pairs, prefix, :reward, result.reward[player])
    add_scalar_metric!(pairs, prefix, :mean_steps, result.mean_steps)
    add_scalar_metric!(pairs, prefix, :detection_rate, result.detected_rate)
    add_scalar_metric!(pairs, prefix, :target_escaped_rate, result.target_escaped_rate)
    add_scalar_metric!(pairs, prefix, :observer_lost_rate, result.observer_lost_rate)
    extra_prefix = replace(prefix, "eval/" => "eval_extra/"; count=1)
    add_scalar_metric!(pairs, extra_prefix, :stderr_reward, result.stderr_reward[player])
    return (; pairs...)
end

struct StatRolloutEvalCallback{G,S,W}
    game::G
    initialstates::S
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
        initialstates=cb.initialstates,
        show_progress=false,
        proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), SDAOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(
                :reward;
                name=:stderr_reward,
                init=zero(MarkovGames.reward_type(cb.game)),
            ),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
    )
end

function (cb::StatRolloutEvalCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    iszero(mod(info.iter, cb.eval_every)) || return

    # Evaluation is the deployed policy: no forced exploration. Search already
    # used epsilon for traversal while accumulating an unperturbed strategy.
    transfer_search = build_search(
        info.oracle;
        epsilon=_ -> EVAL_EPSILON,
        prior_scale=inference_prior_scale,
        queries=inference_tree_queries,
    )
    no_transfer_search = build_search(
        info.oracle;
        epsilon=_ -> EVAL_EPSILON,
        prior_scale=0.0,
        queries=inference_tree_queries,
    )
    transfer_planner = AZ.AlphaZeroPlanner(cb.game, transfer_search)
    no_transfer_planner = AZ.AlphaZeroPlanner(cb.game, no_transfer_search)
    no_burn_1 = Tools.sda_no_burn_heuristic(cb.game, 1)
    no_burn_2 = Tools.sda_no_burn_heuristic(cb.game, 2)
    raw_joint = Tools.OracleStrategyPolicy(cb.game, info.oracle)

    transfer_observer = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(transfer_planner, 1),
        no_burn_2,
    )
    transfer_target = Tools.JointPolicy(
        no_burn_1,
        Tools.SinglePlayerAlphaZeroPolicy(transfer_planner, 2),
    )
    no_transfer_observer = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(no_transfer_planner, 1),
        no_burn_2,
    )
    no_transfer_target = Tools.JointPolicy(
        no_burn_1,
        Tools.SinglePlayerAlphaZeroPolicy(no_transfer_planner, 2),
    )
    raw_observer = Tools.JointPolicy(
        Tools.ProjectedPlayerPolicy(raw_joint, 1),
        no_burn_2,
    )
    raw_target = Tools.JointPolicy(
        no_burn_1,
        Tools.ProjectedPlayerPolicy(raw_joint, 2),
    )

    seed = SEED + 10_000_000 + 4 * info.iter
    Random.seed!(seed)
    transfer_observer_result = rollout_eval(cb, transfer_observer)
    Random.seed!(seed + 1)
    transfer_target_result = rollout_eval(cb, transfer_target)
    Random.seed!(seed)
    no_transfer_observer_result = rollout_eval(cb, no_transfer_observer)
    Random.seed!(seed + 1)
    no_transfer_target_result = rollout_eval(cb, no_transfer_target)
    Random.seed!(seed + 2)
    raw_observer_result = rollout_eval(cb, raw_observer)
    Random.seed!(seed + 3)
    raw_target_result = rollout_eval(cb, raw_target)

    metrics = merge(
        prefixed_eval_metrics(
            transfer_observer_result,
            "eval/transfer_search_observer_vs_no_burn",
            1,
        ),
        prefixed_eval_metrics(
            transfer_target_result,
            "eval/no_burn_vs_transfer_search_target",
            2,
        ),
        prefixed_eval_metrics(
            no_transfer_observer_result,
            "eval/no_transfer_search_observer_vs_no_burn",
            1,
        ),
        prefixed_eval_metrics(
            no_transfer_target_result,
            "eval/no_burn_vs_no_transfer_search_target",
            2,
        ),
        prefixed_eval_metrics(
            raw_observer_result,
            "eval/raw_strategy_observer_vs_no_burn",
            1,
        ),
        prefixed_eval_metrics(
            raw_target_result,
            "eval/no_burn_vs_raw_strategy_target",
            2,
        ),
    )
    println(
        "eval iter $(info.iter): ",
        "transfer observer=$(round(transfer_observer_result.reward[1]; digits=3)) ",
        "target=$(round(transfer_target_result.reward[2]; digits=3)); ",
        "no-transfer observer=$(round(no_transfer_observer_result.reward[1]; digits=3)) ",
        "target=$(round(no_transfer_target_result.reward[2]; digits=3)); ",
        "raw observer=$(round(raw_observer_result.reward[1]; digits=3)) ",
        "target=$(round(raw_target_result.reward[2]; digits=3))",
    )
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

# A fixed state bank and explicit per-evaluation RNG seeds make every checkpoint
# comparison use the same orbital configurations and reproducible search noise.
eval_rng = MersenneTwister(1)
eval_initialstates = [rand(eval_rng, initialstate(game)) for _ in 1:eval_runs]
snr_features = map(eval_initialstates) do s
    Float64(MarkovGames.convert_s(Vector{Float32}, s, game)[16])
end
snr_nonfinite = count(x -> !isfinite(x), snr_features)
snr_finite = filter(isfinite, snr_features)
isempty(snr_finite) && error("All fixed-bank SNR features are nonfinite")

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
    rng=MersenneTwister(SEED),
    optimiser = Flux.Optimisers.Adam(lr)
)

wandb_config = Dict{String,Any}(
    "experiment" => EXPERIMENT_NAME,
    "condition" => string(CONDITION),
    "seed" => SEED,
    "game" => "SNRGameSimple",
    "game/sdagames_tree_sha" => SDAGAMES_TREE_SHA,
    "game/state_features" => 16,
    "game/action_counts" => "$(na1)x$(na2)",
    "search/name" => SEARCH_NAME,
    "search/type" => "MCTSSearch",
    "search/training_tree_queries" => search.tree_queries,
    "search/max_depth" => search.max_depth,
    "search/max_time" => search.max_time,
    "search/backup" => string(search.search_style.backup),
    "search/method" => "plus",
    "search/value_target" => string(search.value_target),
    "search/training_prior_scale" => search.prior_scale,
    "inference/prior_scale" => inference_prior_scale,
    "inference/tree_queries" => inference_tree_queries,
    "inference/equivalent_transfer_weight" => inference_prior_scale / tree_queries,
    "sim_depth" => sim_depth,
    "max_steps" => max_steps,
    "num_steps" => num_steps,
    "update_epochs" => update_epochs,
    "num_batches" => num_batches,
    "width" => width,
    "oracle/shared_trunk" => false,
    "oracle/state_network" => "16-$(width)-$(width)-$(width)-output",
    "oracle/value_weight" => oracle.value_weight,
    "oracle/regret_weight" => oracle.regret_weight,
    "oracle/strategy_weight" => oracle.strategy_weight,
    "critic_type" => critic_type,
    "lr" => lr,
    "lr_decay" => lr_decay,
    "lr_min" => lr_min,
    "lr_max" => lr_max,
    "ema" => ema,
    "ema_decay" => ema_decay,
    "gae_lambda" => gae_lambda,
    "optimizer" => "Adam",
    "epsilon_initial" => epsilon_schedule(1),
    "epsilon_decay" => epsilon_decay,
    "epsilon_min" => epsilon_schedule(typemax(Int)),
    "eval_epsilon" => EVAL_EPSILON,
    "eval_runs" => eval_runs,
    "eval_every" => eval_every,
    "eval/fixed_state_bank_seed" => 1,
    "state_feature/snr_nonfinite" => snr_nonfinite,
    "state_feature/snr_min" => minimum(snr_finite),
    "state_feature/snr_median" => median(snr_finite),
    "state_feature/snr_p95" => quantile(snr_finite, 0.95),
    "state_feature/snr_max" => maximum(snr_finite),
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
    eval_initialstates,
    eval_runs,
    sim_depth,
    eval_every,
    wandb_cb,
)
callbacks = isnothing(wandb_cb) ?
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), stat_eval_cb) :
    (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), wandb_cb, stat_eval_cb)

solve(solver, game; s0=initialstate(game), cb=callbacks)
isnothing(wandb_cb) || close(wandb_cb)
rmprocs(p)
