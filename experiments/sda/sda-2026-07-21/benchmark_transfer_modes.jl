# Compare regret-transfer variants on SDA under both metrics that matter:
#
#   1. exact truncated best-response gap (an exploitability lower bound computed
#      by full game-tree enumeration, replacing the PPO response proxy);
#   2. seat-balanced head-to-head cross-play against the value-only solver.
#
#   julia --project=experiments experiments/sda/sda-2026-07-21/benchmark_transfer_modes.jl \
#       --roots 24 --horizon 4 --episodes 200
include(joinpath(@__DIR__, "transfer_mode_common.jl"))

using DelimitedFiles

const DEFAULT_SOLVERS = [
    "zero_oracle", "value_oracle",
    "warmstart", "capped", "gated",
    "masked_warmstart", "masked_gated", "masked_gated_lemma",
]

function parse_args(args)
    cfg = Dict(
        "roots" => "24", "horizon" => "3", "nreps" => "2", "episodes" => "200",
        "queries" => "100", "max-steps" => "50", "seed" => "20260730",
        "solvers" => join(DEFAULT_SOLVERS, ","), "iter" => "latest",
        "output" => joinpath(@__DIR__, "transfer_mode_benchmark"),
        "skip-gap" => "true", "skip-h2h" => "false", "skip-pool" => "false",
        "update" => "sampled",
    )
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        opt = key[3:end]
        haskey(cfg, opt) || error("Unknown option $(key)")
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        cfg[opt] = args[i]
        i += 1
    end
    return (;
        roots=parse(Int, cfg["roots"]),
        horizon=parse(Int, cfg["horizon"]),
        nreps=parse(Int, cfg["nreps"]),
        episodes=parse(Int, cfg["episodes"]),
        queries=parse(Int, cfg["queries"]),
        max_steps=parse(Int, cfg["max-steps"]),
        seed=parse(Int, cfg["seed"]),
        solvers=String.(split(cfg["solvers"], ",")),
        iter=cfg["iter"],
        output=cfg["output"],
        skip_gap=parse(Bool, cfg["skip-gap"]),
        skip_h2h=parse(Bool, cfg["skip-h2h"]),
        skip_pool=parse(Bool, cfg["skip-pool"]),
        update=Symbol(cfg["update"]),
    )
end

function main(args)
    opts = parse_args(args)
    game = make_game()
    oracles = load_oracles(game, opts.iter)
    # Deployment keeps prior mass at 5% of the online query budget.
    cfg = default_config(;
        tree_queries=opts.queries, prior_scale=0.05 * opts.queries, update=opts.update,
    )
    @printf(
        "checkpoint iter=%d  queries=%d  prior_scale=%.2f  cap_ratio=%.2f  gate_tol=%.2f  update=%s\n",
        oracles.iter, cfg.tree_queries, cfg.prior_scale, cfg.cap_ratio, cfg.gate_tol, cfg.update,
    )

    searches = Dict(n => build_solver(n, game, oracles, cfg) for n in opts.solvers)

    Random.seed!(opts.seed)
    dist = core_initialstate_distribution(game)
    gap_states = [rand(dist) for _ in 1:opts.roots]
    Random.seed!(opts.seed + 1)
    h2h_states = [rand(dist) for _ in 1:opts.episodes]

    rows = []
    pooled = Dict{String,Vector{Float64}}()
    seat_pooled = Dict{String,Dict{Int,Vector{Float64}}}()
    for name in opts.solvers
        search = searches[name]
        gap = gain1 = gain2 = self = NaN
        gap_se = NaN
        if !opts.skip_gap
            t0 = time()
            results = map(enumerate(gap_states)) do (k, s)
                Random.seed!(opts.seed + 7919 * k)
                truncated_gap(game, search, s; horizon=opts.horizon, nreps=opts.nreps, ϵ=cfg.epsilon)
            end
            gap = mean(r -> r.gap, results)
            gap_se = std(map(r -> r.gap, results)) / sqrt(length(results))
            gain1 = mean(r -> r.gain1, results)
            gain2 = mean(r -> r.gain2, results)
            self = mean(r -> r.self, results)
            @printf(
                "[gap] %-20s H=%d gap=%.4f ± %.4f  gain1=%.4f gain2=%.4f self=%.4f  (%.0fs)\n",
                name, opts.horizon, gap, gap_se, gain1, gain2, self, time() - t0,
            )
            flush(stdout)
        end
        h2h = NaN
        h2h_se = NaN
        if !opts.skip_h2h && name != "value_oracle"
            t0 = time()
            r = seat_balanced(
                game, search, searches["value_oracle"], h2h_states;
                max_steps=opts.max_steps, ϵ=cfg.epsilon, seed=opts.seed + 31,
            )
            h2h = r.balanced
            h2h_se = r.se
            @printf(
                "[h2h] %-20s vs value_oracle: %+.4f ± %.4f  (p1 %+.3f, p2 %+.3f) (%.0fs)\n",
                name, h2h, h2h_se, r.as_p1, r.as_p2, time() - t0,
            )
            flush(stdout)
        end
        pool_mean = pool_worst = NaN
        seat1 = seat2 = seat1_worst = seat2_worst = NaN
        pool_detail = Dict{String,Float64}()
        if !opts.skip_pool
            t0 = time()
            r = exploiter_pool_utility(
                game, search, h2h_states;
                max_steps=opts.max_steps, ϵ=cfg.epsilon, seed=opts.seed + 57,
            )
            pool_mean = r.pool_mean
            pool_worst = r.pool_worst
            pool_detail = r.per_member
            pooled[name] = r.pooled
            seat_pooled[name] = r.seat_pooled
            seat1 = get(r.per_seat, 1, NaN)
            seat2 = get(r.per_seat, 2, NaN)
            seat1_worst = get(r.per_seat_worst, 1, NaN)
            seat2_worst = get(r.per_seat_worst, 2, NaN)
            @printf(
                "[pool] %-20s seat1(secures)=%+.4f seat2(concedes)=%+.4f | sum mean=%+.4f worst=%+.4f  %s  (%.0fs)\n",
                name, seat1, seat2, pool_mean, pool_worst,
                join(["$(k)=$(round(v; digits=3))" for (k, v) in sort(collect(pool_detail))], " "),
                time() - t0,
            )
            flush(stdout)
        end
        push!(rows, (;
            solver=name, gap, gap_se, gain1, gain2, self,
            pool_seat1=seat1, pool_seat2=seat2,
            pool_seat1_worst=seat1_worst, pool_seat2_worst=seat2_worst,
            pool_mean, pool_worst,
            pool_vs_zero=get(pool_detail, "zero_oracle", NaN),
            pool_vs_value=get(pool_detail, "value_oracle", NaN),
            pool_vs_full=get(pool_detail, "full_solver", NaN),
            h2h_vs_value=h2h, h2h_se,
            horizon=opts.horizon, nreps=opts.nreps, roots=opts.roots,
            episodes=opts.episodes, queries=cfg.tree_queries,
            prior_scale=cfg.prior_scale, cap_ratio=cfg.cap_ratio,
            gate_tol=cfg.gate_tol, max_steps=opts.max_steps, update=cfg.update,
            checkpoint_iter=oracles.iter,
        ))
    end

    println("\n=== summary ===")
    reference = "value_oracle"
    has_ref = haskey(pooled, reference)
    println("Per-seat best-response utility, both signed so higher is better for the")
    println("solver: seat 1 is the value it secures as player 1, seat 2 the negated")
    println("value it concedes as player 2. A Nash profile requires BOTH to be optimal,")
    println("so both paired deltas must be nonnegative — a NashConv-style sum can hide")
    println("a transfer that helps one seat and hurts the other.")
    println()
    @printf("%-22s %11s %11s %22s %22s %18s\n",
            "solver", "seat1 (↑)", "seat2 (↑)",
            "paired Δ seat1 (↑)", "paired Δ seat2 (↑)", "h2h vs value (↑)")
    for r in rows
        function dtxt(seat)
            (has_ref && haskey(seat_pooled, r.solver) && haskey(seat_pooled, reference) &&
             haskey(seat_pooled[r.solver], seat) && haskey(seat_pooled[reference], seat)) ||
                return "        n/a         "
            d = paired_delta(seat_pooled[r.solver][seat], seat_pooled[reference][seat])
            return @sprintf("%+9.4f ± %-9.4f", d.delta, d.se)
        end
        @printf("%-22s %11.4f %11.4f %22s %22s %8.4f ± %-8.4f\n",
                r.solver, r.pool_seat1, r.pool_seat2, dtxt(1), dtxt(2),
                r.h2h_vs_value, r.h2h_se)
    end
    println()
    @printf("%-22s %14s %14s %24s\n",
            "solver", "sum mean (↑)", "sum worst (↑)", "paired Δ sum vs value (↑)")
    for r in rows
        dsum = "        n/a         "
        if has_ref && haskey(pooled, r.solver)
            d = paired_delta(pooled[r.solver], pooled[reference])
            dsum = @sprintf("%+9.4f ± %-9.4f", d.delta, d.se)
        end
        @printf("%-22s %14.4f %14.4f %24s\n",
                r.solver, r.pool_mean, r.pool_worst, dsum)
    end

    mkpath(opts.output)
    path = joinpath(opts.output, "results.csv")
    columns = propertynames(first(rows))
    writedlm(
        path,
        [reshape(collect(String.(columns)), 1, :);
         reduce(vcat, [reshape(Any[getproperty(r, c) for c in columns], 1, :) for r in rows])],
        ',',
    )
    println("\nwrote ", path)
end

main(ARGS)
