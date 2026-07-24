using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributions
using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using Printf
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

include(joinpath(@__DIR__, "benchmark_regret_transfer_schemes.jl"))

const DEFAULT_HEAD_TO_HEAD_OUTPUT_DIR = joinpath(
    @__DIR__,
    "value_vs_regret_head_to_head",
)

function evaluate_head_to_head(
        game,
        initialstates,
        baseline_oracle,
        hurdle_regressors,
        regret_player;
        max_steps,
        queries,
        max_depth,
        search_epsilon,
        seed,
    )
    value_config = candidate("value_only"; scale=0.0, regret_weight=0.0)
    regret_config = candidate(
        "regret_only_raw_scale5";
        scale=5.0,
        regret_weight=1.0,
        strategy_weight=0.0,
        statistic_weight=0.0,
        reach_power=1.0,
    )
    value_oracle = TransformedRegretOracle(
        baseline_oracle,
        hurdle_regressors,
        :baseline,
        :raw,
        0.0,
        1.0,
        Inf,
    )
    regret_oracle = TransformedRegretOracle(
        baseline_oracle,
        hurdle_regressors,
        :baseline,
        :raw,
        0.0,
        1.0,
        Inf,
    )
    value_search = build_search(
        value_oracle,
        value_config;
        queries,
        max_depth,
        search_epsilon,
    )
    regret_search = build_search(
        regret_oracle,
        regret_config;
        queries,
        max_depth,
        search_epsilon,
    )
    value_planner = AZ.AlphaZeroPlanner(game, value_search)
    regret_planner = AZ.AlphaZeroPlanner(game, regret_search)
    value_player = 3 - regret_player
    regret_policy = Tools.SinglePlayerAlphaZeroPolicy(regret_planner, regret_player)
    value_policy = Tools.SinglePlayerAlphaZeroPolicy(value_planner, value_player)
    joint_policy = regret_player == 1 ?
        Tools.JointPolicy(regret_policy, value_policy) :
        Tools.JointPolicy(value_policy, regret_policy)

    @printf(
        "[head-to-head] start regret_player=%d value_player=%d runs=%d\n",
        regret_player,
        value_player,
        length(initialstates),
    )
    flush(stdout)
    Random.seed!(seed)
    elapsed = @elapsed result = Tools.evaluate_joint_policy(
        game,
        joint_policy,
        length(initialstates);
        max_steps,
        initialstates,
        show_progress=true,
        proc_warn=false,
        parallel=false,
        accumulators=(StepCount(), SDAOutcome()),
        batch_accumulators=(
            MeanResult(:steps; name=:mean_steps),
            Tools.StdErrResult(
                :reward;
                name=:stderr_reward,
                init=zero(MarkovGames.reward_type(game)),
            ),
            RateResult(:detected),
            RateResult(:target_escaped),
            RateResult(:observer_lost),
        ),
    )
    regret_reward = result.reward[regret_player]
    regret_stderr = result.stderr_reward[regret_player]
    @printf(
        "[head-to-head] done regret_player=%d reward=%.6f stderr=%.6f elapsed=%.1fs\n",
        regret_player,
        regret_reward,
        regret_stderr,
        elapsed,
    )
    flush(stdout)
    return (;
        regret_player,
        value_player,
        runs=length(initialstates),
        player1_reward=result.reward[1],
        player1_stderr=result.stderr_reward[1],
        player2_reward=result.reward[2],
        player2_stderr=result.stderr_reward[2],
        regret_only_reward=regret_reward,
        regret_only_stderr=regret_stderr,
        value_only_reward=result.reward[value_player],
        value_only_stderr=result.stderr_reward[value_player],
        mean_steps=result.mean_steps,
        detection_rate=result.detected_rate,
        target_escaped_rate=result.target_escaped_rate,
        observer_lost_rate=result.observer_lost_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        regret_prior_scale=regret_config.scale,
        regret_prior_weight=regret_config.regret_weight,
        strategy_prior_weight=regret_config.strategy_weight,
        statistic_prior_weight=regret_config.statistic_weight,
        prior_reach_power=regret_config.reach_power,
        elapsed_seconds=elapsed,
    )
end

function main(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 500, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 100, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260724, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    output_dir = abspath(option_value(
        args,
        "--output-dir",
        DEFAULT_HEAD_TO_HEAD_OUTPUT_DIR,
        String,
    ))
    fitted_models_path = abspath(option_value(
        args,
        "--fitted-models",
        DEFAULT_FITTED_MODELS,
        String,
    ))
    if test
        runs = min(runs, 2)
        queries = min(queries, 2)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    max_steps > 0 || error("--max-steps must be positive")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")
    isfile(fitted_models_path) || error("Missing fitted models $(fitted_models_path)")

    checkpoint = select_checkpoint(checkpoint_spec)
    oracle_file = joinpath(@__DIR__, "oracle_$(SEARCH_NAME).jld2")
    online_oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(online_oracle, checkpoint)
    baseline_oracle, hurdle_regressors, metadata =
        build_refit_oracles(online_oracle, fitted_models_path)

    game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
    initial_rng = MersenneTwister(seed)
    initial_distribution = core_initialstate_distribution(game)
    initialstates = [rand(initial_rng, initial_distribution) for _ in 1:runs]

    println("[head-to-head] checkpoint=$(checkpoint)")
    println("[head-to-head] fitted_models=$(fitted_models_path)")
    baseline_activation = metadata["baseline_activation"]
    fit_epochs = metadata["epochs"]
    println(
        "[head-to-head] baseline_activation=$(baseline_activation) ",
        "fit_epochs=$(fit_epochs)",
    )
    println(
        "[head-to-head] runs=$(runs) queries=$(queries) depth=$(max_depth) ",
        "steps=$(max_steps) search_epsilon=$(search_epsilon) action_epsilon=0.0",
    )
    flush(stdout)

    rows = map(1:2) do regret_player
        evaluate_head_to_head(
            game,
            initialstates,
            baseline_oracle,
            hurdle_regressors,
            regret_player;
            max_steps,
            queries,
            max_depth,
            search_epsilon,
            seed=seed + regret_player,
        )
    end
    mkpath(output_dir)
    output = write_csv(joinpath(output_dir, "matchups.csv"), rows)
    seat_balanced_reward = sum(row.regret_only_reward for row in rows) / 2
    seat_balanced_stderr = sqrt(
        sum(row.regret_only_stderr^2 for row in rows),
    ) / 2
    summary = (;
        runs_per_orientation=runs,
        total_rollouts=2 * runs,
        seat_balanced_regret_reward=seat_balanced_reward,
        approximate_stderr=seat_balanced_stderr,
        approximate_z=seat_balanced_reward / seat_balanced_stderr,
        approximate_ci95_lower=seat_balanced_reward - 1.96 * seat_balanced_stderr,
        approximate_ci95_upper=seat_balanced_reward + 1.96 * seat_balanced_stderr,
    )
    summary_output = write_csv(joinpath(output_dir, "summary.csv"), [summary])
    println("[head-to-head] wrote $(output)")
    println("[head-to-head] wrote $(summary_output)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
