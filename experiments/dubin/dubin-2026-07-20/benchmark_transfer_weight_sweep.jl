# Transfer-weight sweep for the learned Dubin priors.
#
# Plays SM-AZ with regret-only transfer at each prior_scale = wT1 against
# SM-AZ(-RT) in both seats for every search budget, using the round-robin
# solver definitions behind Table 2. The transfer weight is reported as
# w = prior_scale / source_queries, where T1 = 500 is the training search budget
# that produced the fitted regret targets. See `cells/` for one CSV per matchup.

include(joinpath(@__DIR__, "benchmark_solver_round_robin.jl"))
include(joinpath(@__DIR__, "..", "..", "transfer_weight_sweep_common.jl"))

using POMDPs

const DEFAULT_SWEEP_OUTPUT_DIR = joinpath(@__DIR__, "transfer_weight_sweep")

# `reference` repeats the fixed Table 2 state. `burnin:K` starts from it and
# applies a uniformly drawn number of uniform-random joint actions in 0:K, which
# keeps evaluation states near the self-play distribution the model was trained
# on while removing the dependence on a single state.
function sweep_initial_states(game, spec, runs, seed)
    spec == "reference" && return fill(initial_dubin_state(), runs)
    startswith(spec, "burnin:") || error("--initial-states must be reference or burnin:K")
    max_burnin = parse(Int, last(split(spec, ':')))
    max_burnin >= 0 || error("burn-in length must be nonnegative")
    rng = MersenneTwister(seed)
    A1, A2 = actions(game)
    states = Any[]
    while length(states) < runs
        s = initial_dubin_state()
        for _ in 1:rand(rng, 0:max_burnin)
            isterminal(game, s) && break
            s = @gen(:sp)(game, s, (rand(rng, A1), rand(rng, A2)), rng)
        end
        isterminal(game, s) || push!(states, s)
    end
    return [s for s in states]
end

function main_sweep(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 1000, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", [25, 50, 100, 200], parse_int_list)
    scales = option_value(
        args, "--scales", [0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
        parse_float_list,
    )
    source_queries = option_value(args, "--source-queries", 500, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20261030, x -> parse(Int, x))
    initial_spec = option_value(args, "--initial-states", "burnin:8", String)
    shard_index, shard_count = parse_shard(option_value(args, "--shard", "1/1", String))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    output_dir = abspath(option_value(args, "--output-dir", DEFAULT_SWEEP_OUTPUT_DIR, String))
    if test
        runs = min(runs, 2)
        queries = [2]
        scales = [0.0, 2.5]
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
        output_dir = joinpath(output_dir, "smoke")
    end

    runs > 0 || error("--runs must be positive")
    all(>(0), queries) || error("--tree-queries must be positive")
    all(>=(0.0), scales) || error("--scales must be nonnegative")
    0.0 in scales || @warn "No scale-0 cells; per-seat deltas will be NaN"
    source_queries > 0 || error("--source-queries must be positive")
    0 < max_steps <= MAX_PPO_STEPS || error("--max-steps must be in 1:$(MAX_PPO_STEPS)")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")
    # Processes split the grid, so keep each one on a single BLAS thread.
    shard_count > 1 && BLAS.set_num_threads(1)

    game = DubinMG(V=(1.0, 1.0))
    learned_oracle, model_iter, checkpoint = load_checkpoint_oracle(checkpoint_spec)
    initialstates = sweep_initial_states(game, initial_spec, runs, seed)
    tag = "dubin-weight-sweep $(shard_index)/$(shard_count)"
    println(
        "[$(tag)] checkpoint=$(checkpoint) iteration=$(model_iter) runs=$(runs) ",
        "queries=$(join(queries, ',')) scales=$(join(scales, ',')) T1=$(source_queries) ",
        "initial_states=$(initial_spec) seed=$(seed)",
    )
    flush(stdout)

    cells = shard_cells(sweep_cells(queries, scales), shard_index, shard_count)
    run_sweep_cells(
        cells, output_dir; tag, source_queries, sweep_seed=seed,
        initial_states=initial_spec, model_iter, checkpoint,
    ) do row_solver, column_solver, cell, cell_seed
        evaluate_round_robin_cell(
            row_solver, column_solver, game, learned_oracle, initialstates;
            queries=cell.queries, max_depth, max_steps, search_epsilon,
            prior_scale=cell.scale, seed=cell_seed,
        )
    end
    summary = summarize_sweep_dir(output_dir)
    println("[$(tag)] wrote $(summary)")
    return summary
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_sweep()
