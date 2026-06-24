using Pkg
Pkg.activate("experiments")

using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_DIR = @__DIR__

struct TransferAblationOracle{O}
    base::O
    use_regret::Bool
    use_strategy::Bool
    action_counts::NTuple{2,Int}
end

AZ.value(oracle::TransferAblationOracle, input) = AZ.value(oracle.base, input)

function AZ.state_regret(oracle::TransferAblationOracle, game, state)
    oracle.use_regret && return AZ.state_regret(oracle.base, game, state)
    return map(n -> zeros(Float32, n), oracle.action_counts)
end

function AZ.state_strategy(oracle::TransferAblationOracle, game, state)
    oracle.use_strategy && return AZ.state_strategy(oracle.base, game, state)
    return map(n -> fill(inv(Float32(n)), n), oracle.action_counts)
end

function parse_args(args)
    runs = 100
    mode = "ablation"
    i = 1
    while i <= length(args)
        if args[i] == "--runs"
            i += 1
            runs = parse(Int, args[i])
        elseif args[i] == "--mode"
            i += 1
            mode = args[i]
        else
            error("usage: [--runs N] [--mode ablation|tau_sweep|head_ablation|head_ablation_peak]")
        end
        i += 1
    end
    return (; runs, mode)
end

checkpoint_path(iter::Int) = joinpath(
    EXPERIMENT_DIR,
    "models_smoos",
    "oracle$(AZ.iter2string(iter)).jld2",
)

function evaluate_condition(
        game,
        s0,
        oracle;
        runs::Int,
        epsilon::Float64,
        tau::Float64,
        transfer_weight::Float64=0.1,
        seed::Int,
    )
    Random.seed!(seed)
    search = AZ.SMOOSSearch(;
        oracle,
        oos_iterations=1000,
        max_depth=5,
        ϵ=_ -> epsilon,
        τ=tau,
        transfer_weight,
    )
    planner = AZ.AlphaZeroPlanner(game, search)
    policy = Tools.JointPolicy(
        Tools.SinglePlayerAlphaZeroPolicy(planner, 1),
        DubinTools.dubin_defender_heuristic(game),
    )
    return Tools.evaluate_joint_policy(
        game,
        policy,
        runs;
        max_steps=50,
        initialstates=fill(s0, runs),
        show_progress=false,
        proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), DubinTools.DubinOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(:reward; name=:stderr_reward, init=zero(MarkovGames.reward_type(game))),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
    )
end

args = parse_args(ARGS)
runs = args.runs
game = DubinMG(V=(1.0, 1.0))
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
oracle = AZ.load_oracle(joinpath(EXPERIMENT_DIR, "oracle_smoos.jld2"))

peak_epsilon = 0.3 * 0.999^(820 - 1)
final_epsilon = 0.1
transfer_tau = 111.11111111111111
conditions = if args.mode == "ablation"
    (
        (name="peak_actual", iter=820, epsilon=peak_epsilon, tau=transfer_tau),
        (name="peak_final_epsilon", iter=820, epsilon=final_epsilon, tau=transfer_tau),
        (name="final_actual", iter=1221, epsilon=final_epsilon, tau=transfer_tau),
        (name="final_peak_epsilon", iter=1221, epsilon=peak_epsilon, tau=transfer_tau),
        (name="peak_no_transfer", iter=820, epsilon=peak_epsilon, tau=0.0),
        (name="final_no_transfer", iter=1221, epsilon=peak_epsilon, tau=0.0),
    )
elseif args.mode == "tau_sweep"
    map((0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, transfer_tau)) do tau
        (name="final_tau_$(tau)", iter=1221, epsilon=final_epsilon, tau)
    end
elseif args.mode == "head_ablation"
    (
        (name="both_heads", iter=1221, epsilon=final_epsilon, tau=transfer_tau, use_regret=true, use_strategy=true),
        (name="regret_only", iter=1221, epsilon=final_epsilon, tau=transfer_tau, use_regret=true, use_strategy=false),
        (name="strategy_only", iter=1221, epsilon=final_epsilon, tau=transfer_tau, use_regret=false, use_strategy=true),
        (name="neither_head", iter=1221, epsilon=final_epsilon, tau=transfer_tau, use_regret=false, use_strategy=false),
    )
elseif args.mode == "head_ablation_peak"
    (
        (name="both_heads", iter=820, epsilon=peak_epsilon, tau=transfer_tau, use_regret=true, use_strategy=true),
        (name="regret_only", iter=820, epsilon=peak_epsilon, tau=transfer_tau, use_regret=true, use_strategy=false),
        (name="strategy_only", iter=820, epsilon=peak_epsilon, tau=transfer_tau, use_regret=false, use_strategy=true),
        (name="neither_head", iter=820, epsilon=peak_epsilon, tau=transfer_tau, use_regret=false, use_strategy=false),
    )
else
    error("unknown mode $(args.mode)")
end

for condition in conditions
    Flux.loadmodel!(oracle, checkpoint_path(condition.iter))
    eval_oracle = if hasproperty(condition, :use_regret)
        TransferAblationOracle(oracle, condition.use_regret, condition.use_strategy, length.(actions(game)))
    else
        oracle
    end
    result = evaluate_condition(
        game,
        s0,
        eval_oracle;
        runs,
        epsilon=condition.epsilon,
        tau=condition.tau,
        seed=20260617,
    )
    println((;
        condition...,
        runs,
        reward=result.reward[1],
        stderr_reward=result.stderr_reward[1],
        attacker_goal_rate=result.attacker_goal_rate,
        tagged_rate=result.tagged_rate,
        timeout_rate=result.timeout_rate,
        mean_steps=result.mean_steps,
    ))
end
