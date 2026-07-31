using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DelimitedFiles
using Random

include(joinpath(@__DIR__, "round_robin_support.jl"))

const ROUND_ROBIN_OUTPUT_DIR = joinpath(
    @__DIR__,
    "solver_round_robin_q50_scale2p5",
)

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function round_robin_config(;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    return (;
        backup=:mean,
        value_target=:search,
        tree_queries=queries,
        max_depth,
        epsilon=search_epsilon,
        prior_scale,
        prior_reach_power=1.0,
    )
end

function evaluate_round_robin_cell(
        row_solver,
        column_solver,
        game,
        learned_oracles,
        initialstates,
        cfg;
        max_steps,
        seed,
        model_iter,
        checkpoint,
    )
    row_planner, row_meta = solver_policy(
        row_solver,
        game,
        learned_oracles,
        cfg,
    )
    column_planner, column_meta = solver_policy(
        column_solver,
        game,
        learned_oracles,
        cfg,
    )
    row_policy = Tools.SinglePlayerAlphaZeroPolicy(row_planner, 1)
    column_policy = Tools.SinglePlayerAlphaZeroPolicy(column_planner, 2)
    joint_policy = Tools.JointPolicy(row_policy, column_policy)

    println(
        "[sda-round-robin] start p1=$(row_solver) ",
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
    println(
        "[sda-round-robin] done p1=$(row_solver) p2=$(column_solver) ",
        "utility=$(round(result.reward[1]; digits=6)) ",
        "stderr=$(round(result.stderr_reward[1]; digits=6)) ",
        "elapsed=$(round(elapsed; digits=1))s",
    )
    flush(stdout)

    return (;
        row_solver,
        column_solver,
        runs=length(initialstates),
        p1_utility=result.reward[1],
        p1_stderr=result.stderr_reward[1],
        p2_utility=result.reward[2],
        p2_stderr=result.stderr_reward[2],
        mean_steps=result.mean_steps,
        detection_rate=result.detected_rate,
        target_escaped_rate=result.target_escaped_rate,
        observer_lost_rate=result.observer_lost_rate,
        queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        max_steps,
        search_epsilon=cfg.epsilon,
        action_epsilon=0.0,
        prior_scale=cfg.prior_scale,
        prior_reach_power=cfg.prior_reach_power,
        regret_prior_weight=row_meta.regret_prior_weight,
        strategy_prior_weight=row_meta.strategy_prior_weight,
        statistic_prior_weight=row_meta.statistic_prior_weight,
        row_oracle_kind=row_meta.oracle_kind,
        column_oracle_kind=column_meta.oracle_kind,
        backup=cfg.backup,
        value_target=cfg.value_target,
        initial_state=CORE_DISTRIBUTION_NAME,
        model_iter,
        checkpoint,
        elapsed_seconds=elapsed,
    )
end

function write_rows(path, rows)
    columns = propertynames(first(rows))
    header = reshape(collect(String.(columns)), 1, :)
    data = reduce(vcat, [
        reshape(Any[getproperty(row, column) for column in columns], 1, :)
        for row in rows
    ])
    mkpath(dirname(path))
    writedlm(path, [header; data], ',')
    return path
end

function ordered_values(rows, property)
    return Dict(
        (row.row_solver, row.column_solver) => getproperty(row, property)
        for row in rows
    )
end

function write_matrix(path, values; first_column="p1_solver")
    header = reshape(
        [first_column; ["opponent_$(solver)" for solver in SOLVERS]],
        1,
        :,
    )
    data = reduce(vcat, [
        reshape(
            Any[row_solver; [
                values[(row_solver, column_solver)]
                for column_solver in SOLVERS
            ]],
            1,
            :,
        )
        for row_solver in SOLVERS
    ])
    writedlm(path, [header; data], ',')
    return path
end

function seat_balanced_values(rows)
    utilities = ordered_values(rows, :p1_utility)
    stderrs = ordered_values(rows, :p1_stderr)
    balanced = Dict{Tuple{String,String},Float64}()
    balanced_stderr = Dict{Tuple{String,String},Float64}()
    for row_solver in SOLVERS, column_solver in SOLVERS
        key = (row_solver, column_solver)
        if row_solver == column_solver
            balanced[key] = 0.0
            balanced_stderr[key] = 0.0
        else
            reverse_key = (column_solver, row_solver)
            balanced[key] = (
                utilities[key] - utilities[reverse_key]
            ) / 2
            balanced_stderr[key] = hypot(
                stderrs[key],
                stderrs[reverse_key],
            ) / 2
        end
    end
    return balanced, balanced_stderr
end

function main_round_robin(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 1000, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 50, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon =
        option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    prior_scale =
        option_value(args, "--prior-scale", 2.5, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260726, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    fitted_models_path = abspath(option_value(
        args,
        "--fitted-models",
        DEFAULT_FITTED_MODELS,
        String,
    ))
    output_dir = abspath(option_value(
        args,
        "--output-dir",
        ROUND_ROBIN_OUTPUT_DIR,
        String,
    ))
    if test
        runs = min(runs, 2)
        queries = min(queries, 2)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
        prior_scale = min(prior_scale, Float64(queries))
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    0 < max_steps <= MAX_EPISODE_STEPS ||
        error("--max-steps must be in 1:$(MAX_EPISODE_STEPS)")
    0 <= search_epsilon <= 1 ||
        error("--search-epsilon must be in [0, 1]")
    0 <= prior_scale <= queries ||
        error("--prior-scale must be in [0, tree_queries]")
    isfile(fitted_models_path) ||
        error("Missing fitted regret models: $(fitted_models_path)")

    game = make_game()
    value_oracle, model_iter, checkpoint =
        load_checkpoint_oracle(checkpoint_spec)
    transfer_oracle = load_regret_refit(value_oracle, fitted_models_path)
    learned_oracles = (; value=value_oracle, transfer=transfer_oracle)
    cfg = round_robin_config(;
        queries,
        max_depth,
        search_epsilon,
        prior_scale,
    )
    initial_rng = MersenneTwister(seed)
    initial_distribution = core_initialstate_distribution(game)
    initialstates = [
        rand(initial_rng, initial_distribution)
        for _ in 1:runs
    ]

    println(
        "[sda-round-robin] checkpoint=$(checkpoint) iteration=$(model_iter) ",
        "runs=$(runs) queries=$(queries) prior_scale=$(prior_scale)",
    )
    println("[sda-round-robin] fitted_models=$(fitted_models_path)")
    println("[sda-round-robin] initial_state=$(CORE_DISTRIBUTION_NAME)")
    flush(stdout)

    rows = NamedTuple[]
    for row_solver in SOLVERS, column_solver in SOLVERS
        push!(rows, evaluate_round_robin_cell(
            row_solver,
            column_solver,
            game,
            learned_oracles,
            initialstates,
            cfg;
            max_steps,
            seed,
            model_iter,
            checkpoint,
        ))
    end

    mkpath(output_dir)
    paths = String[]
    push!(paths, write_rows(joinpath(output_dir, "matchups.csv"), rows))
    utilities = ordered_values(rows, :p1_utility)
    stderrs = ordered_values(rows, :p1_stderr)
    push!(paths, write_matrix(
        joinpath(output_dir, "p1_utilities.csv"),
        utilities,
    ))
    push!(paths, write_matrix(
        joinpath(output_dir, "p1_stderrs.csv"),
        stderrs,
    ))
    balanced, balanced_stderr = seat_balanced_values(rows)
    push!(paths, write_matrix(
        joinpath(output_dir, "seat_balanced_utilities.csv"),
        balanced;
        first_column="row_solver",
    ))
    push!(paths, write_matrix(
        joinpath(output_dir, "seat_balanced_stderrs.csv"),
        balanced_stderr;
        first_column="row_solver",
    ))
    foreach(path -> println("[sda-round-robin] wrote $(path)"), paths)
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_round_robin()
