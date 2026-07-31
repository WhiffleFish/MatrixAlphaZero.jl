using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DelimitedFiles
using ExperimentTools
using MarkovGames
using MatrixAlphaZero
using POSGModels.Dubin
using Random

include(joinpath(@__DIR__, "round_robin_support.jl"))

const ROUND_ROBIN_SOLVERS = (
    "zero_oracle",
    "value_oracle",
    "full_solver",
)
const DEFAULT_ROUND_ROBIN_OUTPUT_DIR = joinpath(
    @__DIR__,
    "solver_round_robin_q50_scale2p5",
)

function round_robin_search(
        solver,
        game,
        learned_oracle;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    if solver == "zero_oracle"
        oracle = Tools.ZeroSearchOracle(game)
        candidate = transfer_candidate("zero_oracle"; backup=:mean)
    elseif solver == "value_oracle"
        oracle = ValueOnlySearchOracle(game, learned_oracle)
        candidate = transfer_candidate("value_only_mean"; backup=:mean)
    elseif solver == "full_solver"
        oracle = learned_oracle
        candidate = transfer_candidate(
            "regret_only_mean_s$(prior_scale)";
            scale=prior_scale,
            backup=:mean,
            regret_weight=1.0,
            strategy_weight=0.0,
            statistic_weight=0.0,
            reach_power=1.0,
        )
    else
        error("Unknown round-robin solver $(solver)")
    end
    return build_transfer_search(
        oracle,
        candidate;
        queries,
        max_depth,
        search_epsilon,
    )
end

function evaluate_round_robin_cell(
        row_solver,
        column_solver,
        game,
        learned_oracle,
        initialstates;
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        prior_scale,
        seed,
    )
    p1_search = round_robin_search(
        row_solver,
        game,
        learned_oracle;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    p2_search = round_robin_search(
        column_solver,
        game,
        learned_oracle;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    p1_policy = Tools.SinglePlayerAlphaZeroPolicy(
        AZ.AlphaZeroPlanner(game, p1_search),
        1,
    )
    p2_policy = Tools.SinglePlayerAlphaZeroPolicy(
        AZ.AlphaZeroPlanner(game, p2_search),
        2,
    )
    joint_policy = Tools.JointPolicy(p1_policy, p2_policy)

    println(
        "[dubin-round-robin] start p1=$(row_solver) ",
        "p2=$(column_solver) runs=$(length(initialstates))",
    )
    flush(stdout)
    Random.seed!(seed)
    start_time = time()
    result = rollout_eval(
        game,
        joint_policy,
        initialstates;
        runs=length(initialstates),
        max_steps,
    )
    elapsed = time() - start_time
    utility = result.reward[1]
    stderr = result.stderr_reward[1]
    println(
        "[dubin-round-robin] done p1=$(row_solver) p2=$(column_solver) ",
        "utility=$(round(utility; digits=6)) ",
        "stderr=$(round(stderr; digits=6)) ",
        "elapsed=$(round(elapsed; digits=1))s",
    )
    flush(stdout)
    return (;
        row_solver,
        column_solver,
        runs=length(initialstates),
        p1_utility=utility,
        p1_stderr=stderr,
        p2_utility=result.reward[2],
        p2_stderr=result.stderr_reward[2],
        mean_steps=result.mean_steps,
        attacker_goal_rate=result.attacker_goal_rate,
        tagged_rate=result.tagged_rate,
        timeout_rate=result.timeout_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        prior_scale,
        regret_prior_weight=1.0,
        strategy_prior_weight=0.0,
        statistic_prior_weight=0.0,
        prior_reach_power=1.0,
        backup=:mean,
        elapsed_seconds=elapsed,
    )
end

function write_round_robin_matrix(path, rows, property)
    values = Dict(
        (row.row_solver, row.column_solver) => getproperty(row, property)
        for row in rows
    )
    header = reshape(["p1_solver"; ["p2_$(s)" for s in ROUND_ROBIN_SOLVERS]], 1, :)
    data = reduce(vcat, [
        reshape(
            Any[row_solver; [
                values[(row_solver, column_solver)]
                for column_solver in ROUND_ROBIN_SOLVERS
            ]],
            1,
            :,
        )
        for row_solver in ROUND_ROBIN_SOLVERS
    ])
    writedlm(path, [header; data], ',')
    return path
end

function main_round_robin(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 1000, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 50, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon =
        option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    prior_scale = option_value(args, "--prior-scale", 2.5, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260726, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    output_dir = abspath(option_value(
        args,
        "--output-dir",
        DEFAULT_ROUND_ROBIN_OUTPUT_DIR,
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
    0 < max_steps <= MAX_EPISODE_STEPS ||
        error("--max-steps must be in 1:$(MAX_EPISODE_STEPS)")
    0 <= search_epsilon <= 1 ||
        error("--search-epsilon must be in [0, 1]")
    prior_scale >= 0 || error("--prior-scale must be nonnegative")

    game = DubinMG(V=(1.0, 1.0))
    learned_oracle, model_iter, checkpoint =
        load_checkpoint_oracle(checkpoint_spec)
    initialstates = fill(initial_dubin_state(), runs)
    println(
        "[dubin-round-robin] checkpoint=$(checkpoint) iteration=$(model_iter) ",
        "runs=$(runs) queries=$(queries) prior_scale=$(prior_scale)",
    )
    flush(stdout)

    rows = NamedTuple[]
    for row_solver in ROUND_ROBIN_SOLVERS
        for column_solver in ROUND_ROBIN_SOLVERS
            push!(rows, evaluate_round_robin_cell(
                row_solver,
                column_solver,
                game,
                learned_oracle,
                initialstates;
                queries,
                max_depth,
                max_steps,
                search_epsilon,
                prior_scale,
                seed,
            ))
        end
    end

    mkpath(output_dir)
    detailed_path = write_csv(joinpath(output_dir, "matchups.csv"), rows)
    utility_path = write_round_robin_matrix(
        joinpath(output_dir, "p1_utilities.csv"),
        rows,
        :p1_utility,
    )
    stderr_path = write_round_robin_matrix(
        joinpath(output_dir, "p1_stderrs.csv"),
        rows,
        :p1_stderr,
    )
    println("[dubin-round-robin] wrote $(detailed_path)")
    println("[dubin-round-robin] wrote $(utility_path)")
    println("[dubin-round-robin] wrote $(stderr_path)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_round_robin()
