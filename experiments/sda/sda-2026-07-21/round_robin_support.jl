using Distributions
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

include(joinpath(@__DIR__, "initial_state.jl"))

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const EXPERIMENT_DIR = @__DIR__
const SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const DEFAULT_FITTED_MODELS = joinpath(
    EXPERIMENT_DIR,
    "regret_fit_results_softplus_long",
    "models.jld2",
)
const SOLVERS = ("zero_oracle", "value_oracle", "full_solver")
const MAX_EPISODE_STEPS = 50

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

make_game() = SNRGameSimple(altitude_bounds=(100e3, 2e7))

function checkpoint_iteration(path::AbstractString)
    result = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(result) && error("Cannot parse checkpoint iteration from $(path)")
    return parse(Int, result.captures[1])
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

softplus_refit_output(x) = Flux.softplus.(x)

function with_softplus_output(actor)
    actor isa Chain || error("Expected checkpoint regret actor to be a Flux.Chain")
    return Chain(actor.layers..., softplus_refit_output)
end

function load_regret_refit(online_oracle, fitted_models_path=DEFAULT_FITTED_MODELS)
    isfile(fitted_models_path) || error("Missing fitted regret models: $(fitted_models_path)")
    fitted = JLD2.load(fitted_models_path)
    refit = AZ.FittedRegretModel(
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
    Flux.loadmodel!(refit.regret_head[1], fitted["baseline_p1_state"])
    Flux.loadmodel!(refit.regret_head[2], fitted["baseline_p2_state"])
    return refit
end

function base_search(oracle, cfg)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        search_style=AZ.RegretMatchingSearch(; backup=cfg.backup, method=AZ.Plus()),
        value_target=cfg.value_target,
        ϵ=_ -> cfg.epsilon,
    )
end

function full_search(oracle, cfg)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        search_style=AZ.RegretMatchingSearch(; backup=cfg.backup, method=AZ.Plus()),
        value_target=cfg.value_target,
        ϵ=_ -> cfg.epsilon,
        prior_scale=cfg.prior_scale,
        regret_prior_weight=1.0,
        strategy_prior_weight=0.0,
        statistic_prior_weight=0.0,
        prior_reach_power=cfg.prior_reach_power,
    )
end

function solver_policy(solver::AbstractString, game, learned_oracles, cfg)
    if solver == "zero_oracle"
        search = base_search(Tools.ZeroSearchOracle(game), cfg)
        meta = (oracle_kind="uniform_zero", regret_prior_weight=0.0,
            strategy_prior_weight=0.0, statistic_prior_weight=0.0)
    elseif solver == "value_oracle"
        search = base_search(ValueOnlySearchOracle(game, learned_oracles.value), cfg)
        meta = (oracle_kind="learned_value_only", regret_prior_weight=0.0,
            strategy_prior_weight=0.0, statistic_prior_weight=0.0)
    elseif solver == "full_solver"
        search = full_search(learned_oracles.transfer, cfg)
        meta = (oracle_kind="learned_value_regret_only", regret_prior_weight=1.0,
            strategy_prior_weight=0.0, statistic_prior_weight=0.0)
    else
        error("Unsupported solver $(solver)")
    end
    return AZ.AlphaZeroPlanner(game, search), meta
end

function rollout_eval(game, joint_policy, initialstates; runs::Int, max_steps::Int)
    return Tools.evaluate_joint_policy(
        game, joint_policy, runs;
        max_steps, initialstates, show_progress=false, proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), SDAOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(:reward; name=:stderr_reward,
                init=zero(MarkovGames.reward_type(game))),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
    )
end
