using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "benchmark_solver_round_robin.jl"))
include(joinpath(@__DIR__, "..", "..", "ppo_best_response_common.jl"))

# Approximate best responses to the three solvers of the paper's SDA round robin
# (SM-MCTS-A = zero_oracle, SM-AZ (-RT) = value_oracle, SM-AZ = full_solver with the
# refit regret heads), built by `solver_policy` with the round robin's settings. Behavior
# cloning, PPO and evaluation all sample from the correlated LEO initial-state law.

const DEFAULT_BR_OUTPUT_DIR = joinpath(@__DIR__, "ppo_best_response_q50_scale2p5")

function main_best_response(args=ARGS)
    opts = parse_br_options(args, Dict(
        "seed" => "20260721",
        "fitted-models" => DEFAULT_FITTED_MODELS,
        "output-dir" => DEFAULT_BR_OUTPUT_DIR,
    ))
    game = make_game()
    settings = br_settings(opts, game)
    output_dir = abspath(opts["output-dir"])
    settings.test && (output_dir = joinpath(output_dir, "smoke"))
    fitted_models_path = abspath(opts["fitted-models"])
    limit_blas_threads()

    value_oracle, model_iter, checkpoint = load_checkpoint_oracle(opts["checkpoint"])
    transfer_oracle = load_regret_refit(value_oracle, fitted_models_path)
    learned_oracles = (; value=value_oracle, transfer=transfer_oracle)
    search_cfg = round_robin_config(; settings.search...)
    planners = [
        solver => first(solver_policy(solver, game, learned_oracles, search_cfg))
        for solver in BR_SOLVER_NAMES
    ]
    initial_distribution = core_initialstate_distribution(game)
    metadata = merge(br_metadata(settings), (;
        initial_state=CORE_DISTRIBUTION_NAME,
        fitted_models=fitted_models_path,
        model_iter,
        checkpoint,
    ))
    println("[sda-ppo-br] checkpoint=$(checkpoint) threads=$(Threads.nthreads()) ",
        "search=$(settings.search) output=$(output_dir)")
    flush(stdout)

    return Tools.run_ppo_best_response_suite(
        game,
        planners;
        settings.suite...,
        heuristics=player -> ["no_burn" => Tools.sda_no_burn_heuristic(game, player)],
        train_initialstate_dist=initial_distribution,
        eval_initialstate_dist=initial_distribution,
        accumulators=(StepCount(), SDAOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
        output_dir,
        metadata,
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_best_response()
