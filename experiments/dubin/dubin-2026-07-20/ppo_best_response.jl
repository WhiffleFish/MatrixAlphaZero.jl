using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "benchmark_solver_round_robin.jl"))
include(joinpath(@__DIR__, "..", "..", "ppo_best_response_common.jl"))

# Approximate best responses to the three solvers of the paper's Dubins round robin
# (SM-MCTS-A = zero_oracle, SM-AZ (-RT) = value_oracle, SM-AZ = full_solver), built by
# `round_robin_search` with the same settings. Evaluation starts from the round robin's
# fixed reference state. Behavior cloning and PPO train on a mixture of that state and
# the game's uniform initial states, so PPO also sees states a uniform policy rarely
# reaches from the reference start.

const DEFAULT_BR_OUTPUT_DIR = joinpath(@__DIR__, "ppo_best_response_q50_scale2p5")

function main_best_response(args=ARGS)
    opts = parse_br_options(args, Dict(
        "seed" => "20260720",
        "train-reference-fraction" => "0.5",
        "output-dir" => DEFAULT_BR_OUTPUT_DIR,
    ))
    game = DubinMG(V=(1.0, 1.0))
    settings = br_settings(opts, game)
    output_dir = abspath(opts["output-dir"])
    settings.test && (output_dir = joinpath(output_dir, "smoke"))
    reference_fraction = parse(Float64, opts["train-reference-fraction"])
    0 <= reference_fraction <= 1 || error("--train-reference-fraction must be in [0, 1]")
    limit_blas_threads()

    learned_oracle, model_iter, checkpoint = load_checkpoint_oracle(opts["checkpoint"])
    planners = [
        solver => AZ.AlphaZeroPlanner(
            game,
            round_robin_search(solver, game, learned_oracle; settings.search...),
        )
        for solver in BR_SOLVER_NAMES
    ]
    reference = Deterministic(initial_dubin_state())
    train_initialstate_dist = Tools.mixture_initialstate(
        [reference, initialstate(game)],
        [reference_fraction, 1 - reference_fraction],
    )
    metadata = merge(br_metadata(settings), (;
        train_reference_fraction=reference_fraction,
        model_iter,
        checkpoint,
    ))
    println("[dubin-ppo-br] checkpoint=$(checkpoint) threads=$(Threads.nthreads()) ",
        "search=$(settings.search) output=$(output_dir)")
    flush(stdout)

    return Tools.run_ppo_best_response_suite(
        game,
        planners;
        settings.suite...,
        heuristics=player -> [
            (player == 1 ? "attacker" : "defender") => heuristic_player_policy(game, player),
        ],
        train_initialstate_dist,
        eval_initialstate_dist=reference,
        accumulators=(StepCount(), DubinTools.DubinOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            RateResult(:attacker_goal),
            RateResult(:tagged),
            RateResult(:timeout),
        ),
        output_dir,
        metadata,
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_best_response()
