using Pkg
Pkg.activate("experiments")

using ExperimentTools
using Statistics

get(ENV, "WANDB_API_KEY", "") != "" || error(
    "WANDB_API_KEY not set — export it before running this script."
)

println("Fetching runs from W&B…")
runs = fetch_wandb_runs("Matrix AlphaZero")
isempty(runs) && error("No runs found in project 'Matrix AlphaZero'.")
println("Found $(length(runs)) run(s)\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

finite_vec(v) = filter(isfinite, Float64.(v))

function summary(v)
    fv = finite_vec(v)
    isempty(fv) && return (first=NaN, last=NaN, min=NaN, max=NaN, mean=NaN)
    return (
        first = fv[1],
        last  = fv[end],
        min   = minimum(fv),
        max   = maximum(fv),
        mean  = mean(fv),
    )
end

trend(v) = let fv = finite_vec(v)
    length(fv) < 2 ? NaN : fv[end] - fv[1]
end

rpad2(s, n) = rpad(string(s), n)
fmt(x::Real) = isnan(x) ? "     —    " : lpad(round(x; digits=4), 10)

function section(title)
    bar = "═" ^ 70
    println("\n", bar)
    println("  ", title)
    println(bar)
end

function metric_table(run, metrics)
    avail = filter(m -> haskey(run.metrics, m), metrics)
    isempty(avail) && (println("  (no data)"); return)
    header = rpad2("metric", 26) * "   first" * "    last" * "     min" * "     max" * "    mean" * "   trend"
    println("  ", header)
    println("  ", "─" ^ length(header))
    for m in avail
        s = summary(run.metrics[m])
        t = trend(run.metrics[m])
        sign = t > 0 ? "↑" : t < 0 ? "↓" : "→"
        println("  ", rpad2(m, 26),
            fmt(s.first), fmt(s.last), fmt(s.min), fmt(s.max), fmt(s.mean),
            "  ", sign, " ", round(t; digits=4))
    end
end

# ── Per-run analysis ───────────────────────────────────────────────────────────

TRAINING_HEALTH = [
    "mean_loss", "mean_value_loss", "mean_policy_loss",
    "mean_grad_norm", "max_grad_norm",
]

ORACLE_QUALITY = [
    "value_pred_mse",
    "policy_entropy_p1", "policy_entropy_p2",
    "policy_kl_p1", "policy_kl_p2",
    "search_oracle_kl_p1", "search_oracle_kl_p2",
]

SELFPLAY = [
    "mean_ep_length", "mean_reward", "reward_std",
    "buffer_size", "buffer_turnover",
]

for run in runs
    section("Run: $(run.name)  [$(run.state)]  id=$(run.id)")

    if !isempty(run.config)
        println("  Config:")
        for (k, v) in sort(collect(run.config))
            println("    $(rpad(k, 20)) = $v")
        end
    end

    n_steps = isempty(run.metrics) ? 0 :
        maximum(length(v) for v in values(run.metrics))
    println("\n  Training steps logged: $n_steps")

    println("\n  ── Training Health ──")
    metric_table(run, TRAINING_HEALTH)

    println("\n  ── Oracle Quality ──")
    metric_table(run, ORACLE_QUALITY)

    println("\n  ── Self-play / Buffer ──")
    metric_table(run, SELFPLAY)
end

# ── Cross-run comparison (if multiple runs) ────────────────────────────────────

if length(runs) > 1
    section("Cross-run Comparison (final-iteration values)")

    compare_metrics = [
        "mean_loss", "mean_value_loss", "mean_policy_loss",
        "mean_grad_norm", "value_pred_mse",
        "policy_entropy_p1", "search_oracle_kl_p1",
        "mean_reward", "mean_ep_length",
    ]

    name_w = maximum(length(r.name) for r in runs) + 2
    header = rpad2("metric", 28) * join(rpad2(r.name, name_w) for r in runs)
    println("  ", header)
    println("  ", "─" ^ length(header))

    for m in compare_metrics
        row = rpad2(m, 28)
        for run in runs
            val = if haskey(run.metrics, m)
                fv = finite_vec(run.metrics[m])
                isempty(fv) ? NaN : fv[end]
            else
                NaN
            end
            row *= rpad2(isnan(val) ? "—" : string(round(val; digits=4)), name_w)
        end
        println("  ", row)
    end

    # Highlight winners per metric
    println("\n  ── Best per metric ──")
    better_lower = Set([
        "mean_loss", "mean_value_loss", "mean_policy_loss",
        "mean_grad_norm", "max_grad_norm", "value_pred_mse",
        "search_oracle_kl_p1", "search_oracle_kl_p2",
    ])
    for m in compare_metrics
        vals = map(runs) do r
            haskey(r.metrics, m) ? let fv = finite_vec(r.metrics[m])
                isempty(fv) ? NaN : fv[end]
            end : NaN
        end
        finite_idx = findall(v -> !isnan(v), vals)
        isempty(finite_idx) && continue
        best_idx = m in better_lower ?
            finite_idx[argmin(vals[finite_idx])] :
            finite_idx[argmax(vals[finite_idx])]
        println("  $(rpad(m, 28)) → best: $(runs[best_idx].name)  ($(round(vals[best_idx]; digits=4)))")
    end
end

println("\n" * "═" ^ 70)
println("  Analysis complete.")
println("═" ^ 70)
