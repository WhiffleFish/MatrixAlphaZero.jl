# Transfer-weight ablation in the exact Figure 2 setting.
#
# Reuses the network-free finite Dubin tree from
# `tabular_regret_transfer_heatmap.jl` and the state it selected. Every
# nonterminal node is solved as its own root. The source solve runs T1 updates,
# its average regrets/strategies are corrupted by δR = δσ, and the target solve
# is warm-started with prior_scale = wT1 before T2 updates on the real values.
# With w = 1, variant `full`, and T1 = T2 = 64 the seeds match Figure 2, so
# those rows reproduce the published heatmap cells.

include(joinpath(@__DIR__, "tabular_regret_transfer_heatmap.jl"))

const SWEEP_VARIANTS = Dict(
    # Figure 2: regret, average strategy, and pseudo-count statistics.
    "full" => (; regret_weight=1.0, strategy_weight=1.0, statistic_weight=1.0),
    # Deployed SM-AZ: regret transfer only.
    "regret_only" => (; regret_weight=1.0, strategy_weight=0.0, statistic_weight=0.0),
)

function parse_sweep_cli(args)
    cfg = Dict{String,Any}(
        "state_csv" => joinpath(SCRIPT_DIR, "tabular_regret_transfer_results", "selected_state.csv"),
        "depth" => 5,
        "source_iterations" => 64,
        "target_iterations" => [64],
        "weights" => [1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0, 2.0, 4.0],
        "transfer_errors" => [0.0, 0.05, 0.10, 0.20, 0.40],
        "leaf_errors" => [0.0],
        "variants" => ["full", "regret_only"],
        "trials" => 32,
        "epsilon" => 0.10,
        "seed" => 20260719,
        "output" => joinpath(SCRIPT_DIR, "tabular_transfer_weight_results"),
        "test" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test"
            cfg["test"] = true
            i += 1
            continue
        end
        startswith(arg, "--") || error("Unexpected argument $(arg)")
        parts = split(arg[3:end], '='; limit=2)
        key = replace(first(parts), '-' => '_')
        haskey(cfg, key) || error("Unknown option --$(first(parts))")
        value = if length(parts) == 2
            last(parts)
        else
            i += 1
            i <= length(args) || error("Missing value for $(arg)")
            args[i]
        end
        if key in ("output", "state_csv")
            cfg[key] = value
        elseif key == "epsilon"
            cfg[key] = parse(Float64, value)
        elseif key in ("weights", "transfer_errors", "leaf_errors")
            cfg[key] = parse_list(Float64, value)
        elseif key == "target_iterations"
            cfg[key] = parse_list(Int, value)
        elseif key == "variants"
            cfg[key] = String.(strip.(split(value, ',')))
        else
            cfg[key] = parse(Int, value)
        end
        i += 1
    end
    if cfg["test"]
        merge!(cfg, Dict(
            "depth" => 2,
            "source_iterations" => 8,
            "target_iterations" => [8],
            "weights" => [0.5, 1.0],
            "transfer_errors" => [0.0, 0.2],
            "leaf_errors" => [0.0],
            "trials" => 2,
            "output" => joinpath(abspath(cfg["output"]), "smoke"),
        ))
    end
    all(v -> haskey(SWEEP_VARIANTS, v), cfg["variants"]) ||
        error("--variants must be drawn from $(join(sort!(collect(keys(SWEEP_VARIANTS))), ','))")
    cfg["source_iterations"] > 1 || error("--source-iterations must exceed one")
    all(>(1), cfg["target_iterations"]) || error("--target-iterations must exceed one")
    all(>(0), cfg["weights"]) || error("--weights must be positive (w = 0 is the restart baseline)")
    all(e -> 0 <= e <= 0.5, cfg["transfer_errors"]) || error("transfer errors must be in [0, 0.5]")
    all(e -> 0 <= e <= 1.0, cfg["leaf_errors"]) || error("leaf errors must be in [0, 1]")
    cfg["trials"] > 0 || error("--trials must be positive")
    return cfg
end

function load_selected_state(path)
    data, header = readdlm(path, ','; header=true)
    column(name) = Float64(data[1, findfirst(==(name), vec(header))])
    return JointDubinState(
        SA[column("attacker_x"), column("attacker_y"), column("attacker_heading")],
        SA[column("defender_x"), column("defender_y"), column("defender_heading")],
    )
end

# Payoff range of each nonterminal node's exact continuation matrix game.
function node_payoff_ranges(tree::FiniteTree)
    gamma = discount(tree.game)
    return [
        begin
            q = tree.rewards[i] .+ gamma .* tree.values[tree.children[i]]
            maximum(q) - minimum(q)
        end
        for i in eachindex(tree.states) if !isempty(tree.children[i])
    ]
end

# Reviewer 5's sufficient condition from Eq. (9): w ≤ T1Δu²/(Δu√T1 + T1δR)².
r5_weight_threshold(T1, payoff_range, regret_error) =
    T1 * payoff_range^2 / (payoff_range * sqrt(T1) + T1 * regret_error)^2

const TRIAL_HEADER = [
    "variant", "source_iterations", "target_iterations", "leaf_error", "transfer_error",
    "weight", "prior_scale", "trial", "baseline_gap", "transfer_gap", "improvement",
    "achieved_leaf_error", "achieved_regret_error", "achieved_strategy_error",
    "elapsed_seconds",
]

function completed_trials(path)
    isfile(path) || return Set{Int}()
    data, header = readdlm(path, ','; header=true)
    isempty(data) && return Set{Int}()
    return Set{Int}(Int.(data[:, findfirst(==("trial"), vec(header))]))
end

function append_rows(path, rows)
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, join(TRIAL_HEADER, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
    return path
end

# One trial covers the whole grid so that every transferred solve is paired
# with the cold solve that shares its target-search seeds.
function run_trial(tree, cfg, trial)
    T1 = cfg["source_iterations"]
    trial_seed = cfg["seed"] + 1_000_000 * trial
    exact = exact_value_dict(tree)
    exact_oracle = tabular_oracle(tree, exact)
    rows = Vector{Any}[]
    baselines = Dict(
        T2 => exact_nash_gap(
            tree,
            solve_all_nodes(
                tree, exact_oracle, cfg, trial_seed + 10; transfer=false, iterations=T2,
            ).policy...,
        ).gap
        for T2 in cfg["target_iterations"]
    )
    for leaf_error in cfg["leaf_errors"]
        values, achieved_leaf = corrupted_values(tree, leaf_error, trial_seed + 1)
        source = solve_all_nodes(
            tree, tabular_oracle(tree, values), cfg, trial_seed + 20;
            transfer=false, extract_source=true, iterations=T1,
        )
        for transfer_error in cfg["transfer_errors"]
            regrets, strategies, achieved_regret, achieved_strategy = corrupted_priors(
                tree, source.average_regret, source.average_strategy,
                transfer_error, trial_seed + 2,
            )
            oracle = tabular_oracle(tree, exact; regrets, strategies)
            for T2 in cfg["target_iterations"], variant in cfg["variants"], w in cfg["weights"]
                prior_scale = w * T1
                elapsed = @elapsed solved = solve_all_nodes(
                    tree, oracle, cfg, trial_seed + 10;
                    transfer=true, iterations=T2, prior_scale, SWEEP_VARIANTS[variant]...,
                )
                gap = exact_nash_gap(tree, solved.policy...).gap
                push!(rows, Any[
                    variant, T1, T2, leaf_error, transfer_error, w, prior_scale, trial,
                    baselines[T2], gap, baselines[T2] - gap,
                    achieved_leaf, achieved_regret, achieved_strategy, elapsed,
                ])
            end
        end
    end
    return rows
end

function summarize_sweep(trials_path, summary_path, cfg, payoff_ranges)
    data, header = readdlm(trials_path, ','; header=true)
    col = Dict(name => j for (j, name) in enumerate(vec(header)))
    keys_of(r) = (
        String(data[r, col["variant"]]), Int(data[r, col["target_iterations"]]),
        Float64(data[r, col["leaf_error"]]), Float64(data[r, col["transfer_error"]]),
        Float64(data[r, col["weight"]]),
    )
    groups = Dict{Any,Vector{Int}}()
    for r in axes(data, 1)
        push!(get!(groups, keys_of(r), Int[]), r)
    end
    T1 = cfg["source_iterations"]
    root_range = payoff_ranges.root
    positive_range = payoff_ranges.median_positive
    summary_header = [
        "variant", "source_iterations", "target_iterations", "leaf_error", "transfer_error",
        "weight", "prior_scale", "trials", "baseline_gap_mean", "baseline_gap_sem",
        "transfer_gap_mean", "transfer_gap_sem", "improvement_mean", "improvement_sem",
        "r5_threshold_root_range", "r5_threshold_median_positive_range",
    ]
    summary_rows = Vector{Any}[]
    for key in sort!(collect(keys(groups)))
        variant, T2, leaf_error, transfer_error, w = key
        r = groups[key]
        b = Float64.(data[r, col["baseline_gap"]])
        t = Float64.(data[r, col["transfer_gap"]])
        g = Float64.(data[r, col["improvement"]])
        push!(summary_rows, Any[
            variant, T1, T2, leaf_error, transfer_error, w, w * T1, length(r),
            mean(b), sem(b), mean(t), sem(t), mean(g), sem(g),
            r5_weight_threshold(T1, root_range, transfer_error),
            r5_weight_threshold(T1, positive_range, transfer_error),
        ])
    end
    write_csv(summary_path, summary_header, summary_rows)
    return summary_path
end

function main_sweep(args=ARGS)
    cfg = parse_sweep_cli(args)
    output = abspath(cfg["output"])
    mkpath(output)
    inner_game = DubinMG(V=(1.0, 1.0))
    state = load_selected_state(cfg["state_csv"])
    tree = build_tree(inner_game, state, cfg["depth"])
    ranges = node_payoff_ranges(tree)
    # Many nodes have identical children (range 0), so summarize the rest.
    payoff_ranges = (;
        root=first(ranges),
        median_positive=median(filter(>(0), ranges)),
        max=maximum(ranges),
    )
    # Functions shared with the heatmap read the epsilon and seed from cfg.
    cfg["iterations"] = cfg["source_iterations"]

    open(joinpath(output, "config.txt"), "w") do io
        for key in sort!(collect(keys(cfg)))
            println(io, key, "=", cfg[key])
        end
        println(io, "threads=", nthreads())
        println(io, "selected_attacker=", collect(state.attacker))
        println(io, "selected_defender=", collect(state.defender))
        println(io, "finite_tree_nodes=", length(tree.states))
        println(io, "payoff_range_root=", payoff_ranges.root)
        println(io, "payoff_range_median_positive=", payoff_ranges.median_positive)
        println(io, "payoff_range_max=", payoff_ranges.max)
    end

    trials_path = joinpath(output, "trials.csv")
    summary_path = joinpath(output, "summary.csv")
    done = completed_trials(trials_path)
    remaining = [trial for trial in 1:cfg["trials"] if !(trial in done)]
    per_trial = length(cfg["leaf_errors"]) * length(cfg["transfer_errors"]) *
        length(cfg["target_iterations"]) * length(cfg["variants"]) * length(cfg["weights"])
    @printf(
        "[tabular-weight-sweep] threads=%d nodes=%d T1=%d T2=%s rows/trial=%d trials done=%d remaining=%d\n",
        nthreads(), length(tree.states), cfg["source_iterations"],
        join(cfg["target_iterations"], ','), per_trial, length(done), length(remaining),
    )
    @printf(
        "[tabular-weight-sweep] payoff range root=%.4f median(>0)=%.4f max=%.4f\n",
        payoff_ranges.root, payoff_ranges.median_positive, payoff_ranges.max,
    )
    flush(stdout)

    start = time()
    for (k, trial) in enumerate(remaining)
        trial_start = time()
        rows = run_trial(tree, cfg, trial)
        append_rows(trials_path, rows)
        summarize_sweep(trials_path, summary_path, cfg, payoff_ranges)
        per = (time() - start) / k
        @printf(
            "[tabular-weight-sweep] trial %d done in %.1fs (%d/%d, eta %.1f min)\n",
            trial, time() - trial_start, k, length(remaining),
            per * (length(remaining) - k) / 60,
        )
        flush(stdout)
    end
    isfile(trials_path) && summarize_sweep(trials_path, summary_path, cfg, payoff_ranges)
    println("[tabular-weight-sweep] wrote $(summary_path)")
    return summary_path
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_sweep()
