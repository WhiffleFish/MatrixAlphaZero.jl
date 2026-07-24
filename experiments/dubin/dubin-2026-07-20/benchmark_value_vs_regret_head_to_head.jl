using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using Random

include(joinpath(@__DIR__, "benchmark_regret_only_transfer.jl"))

const DEFAULT_HEAD_TO_HEAD_OUTPUT_DIR = joinpath(
    @__DIR__,
    "value_vs_regret_head_to_head",
)

function parse_scales(spec)
    scales = parse.(Float64, split(spec, ','))
    isempty(scales) && error("--scales must contain at least one scale")
    all(>=(0.0), scales) || error("--scales must be nonnegative")
    return scales
end

function evaluate_head_to_head(
        game,
        learned_oracle,
        regret_scale,
        regret_player,
        initialstates;
        max_steps,
        queries,
        max_depth,
        search_epsilon,
        seed,
    )
    value_candidate = transfer_candidate("value_only_mean")
    regret_candidate = transfer_candidate(
        "regret_only_mean_s$(regret_scale)";
        scale=regret_scale,
        backup=:mean,
        regret_weight=1.0,
        strategy_weight=0.0,
        statistic_weight=0.0,
        reach_power=1.0,
    )
    value_oracle = ValueOnlySearchOracle(game, learned_oracle)
    value_search = build_transfer_search(
        value_oracle,
        value_candidate;
        queries,
        max_depth,
        search_epsilon,
    )
    regret_search = build_transfer_search(
        learned_oracle,
        regret_candidate;
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

    println(
        "[dubin-head-to-head] start scale=$(regret_scale) ",
        "regret_player=$(regret_player) runs=$(length(initialstates))",
    )
    flush(stdout)
    Random.seed!(seed + regret_player)
    elapsed = @elapsed result = rollout_eval(
        game,
        joint_policy,
        initialstates;
        runs=length(initialstates),
        max_steps,
    )
    regret_reward = result.reward[regret_player]
    regret_stderr = result.stderr_reward[regret_player]
    println(
        "[dubin-head-to-head] done scale=$(regret_scale) ",
        "regret_player=$(regret_player) reward=$(round(regret_reward; digits=6)) ",
        "stderr=$(round(regret_stderr; digits=6)) elapsed=$(round(elapsed; digits=1))s",
    )
    flush(stdout)
    return (;
        regret_scale,
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
        attacker_goal_rate=result.attacker_goal_rate,
        tagged_rate=result.tagged_rate,
        timeout_rate=result.timeout_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        regret_prior_weight=1.0,
        strategy_prior_weight=0.0,
        statistic_prior_weight=0.0,
        prior_reach_power=1.0,
        backup=:mean,
        elapsed_seconds=elapsed,
    )
end

function main(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 1000, x -> parse(Int, x))
    scales = parse_scales(option_value(args, "--scales", "5,50", String))
    queries = option_value(args, "--tree-queries", 100, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260726, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    output_dir = abspath(option_value(
        args,
        "--output-dir",
        DEFAULT_HEAD_TO_HEAD_OUTPUT_DIR,
        String,
    ))
    if test
        runs = min(runs, 2)
        scales = [5.0]
        queries = min(queries, 2)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    0 < max_steps <= MAX_PPO_STEPS ||
        error("--max-steps must be in 1:$(MAX_PPO_STEPS)")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")

    game = DubinMG(V=(1.0, 1.0))
    learned_oracle, model_iter, checkpoint = load_checkpoint_oracle(checkpoint_spec)
    initialstates = fill(initial_dubin_state(), runs)
    println("[dubin-head-to-head] checkpoint=$(checkpoint) iteration=$(model_iter)")
    println(
        "[dubin-head-to-head] scales=$(join(scales, ',')) runs=$(runs) ",
        "queries=$(queries) depth=$(max_depth) steps=$(max_steps) ",
        "search_epsilon=$(search_epsilon) action_epsilon=0.0",
    )
    flush(stdout)

    rows = NamedTuple[]
    for regret_scale in scales, regret_player in 1:2
        push!(rows, evaluate_head_to_head(
            game,
            learned_oracle,
            regret_scale,
            regret_player,
            initialstates;
            max_steps,
            queries,
            max_depth,
            search_epsilon,
            seed,
        ))
    end
    summaries = map(scales) do regret_scale
        scale_rows = filter(row -> row.regret_scale == regret_scale, rows)
        balanced_reward =
            sum(row.regret_only_reward for row in scale_rows) / length(scale_rows)
        approximate_stderr = sqrt(
            sum(row.regret_only_stderr^2 for row in scale_rows),
        ) / length(scale_rows)
        return (;
            regret_scale,
            runs_per_orientation=runs,
            total_rollouts=2 * runs,
            seat_balanced_regret_reward=balanced_reward,
            approximate_stderr,
            approximate_z=balanced_reward / approximate_stderr,
            approximate_ci95_lower=balanced_reward - 1.96 * approximate_stderr,
            approximate_ci95_upper=balanced_reward + 1.96 * approximate_stderr,
        )
    end

    mkpath(output_dir)
    matchup_output = write_csv(joinpath(output_dir, "matchups.csv"), rows)
    summary_output = write_csv(joinpath(output_dir, "summary.csv"), summaries)
    println("[dubin-head-to-head] wrote $(matchup_output)")
    println("[dubin-head-to-head] wrote $(summary_output)")
    return rows, summaries
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
