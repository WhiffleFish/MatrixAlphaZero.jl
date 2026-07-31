using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_DIR = @__DIR__
const SEARCH_NAME = "rm_plus_no_transfer_train"
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

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function transfer_candidate(
        name;
        scale=0.0,
        backup=:mean,
        regret_weight=0.0,
        strategy_weight=0.0,
        statistic_weight=0.0,
        reach_power=1.0,
    )
    return (;
        name,
        scale=Float64(scale),
        backup=Symbol(backup),
        regret_weight=Float64(regret_weight),
        strategy_weight=Float64(strategy_weight),
        statistic_weight=Float64(statistic_weight),
        reach_power=Float64(reach_power),
    )
end

function build_transfer_search(
        oracle,
        candidate;
        queries,
        max_depth,
        search_epsilon,
    )
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(;
            backup=candidate.backup,
            method=AZ.Plus(),
        ),
        value_target=:search,
        ϵ=_ -> search_epsilon,
        prior_scale=candidate.scale,
        regret_prior_weight=candidate.regret_weight,
        strategy_prior_weight=candidate.strategy_weight,
        statistic_prior_weight=candidate.statistic_weight,
        prior_reach_power=candidate.reach_power,
    )
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
