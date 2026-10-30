using Pkg
Pkg.activate(@__DIR__)

# Figures for the transfer-weight ablation.
#
# (a) Tabular Figure 2 setting: exact Nash-gap reduction from transfer against
#     w for each transfer error δR = δσ, with Reviewer 5's sufficient threshold
#     w ≤ T1Δu²/(Δu√T1 + T1δR)² marked in the matching color.
# (b) Learned priors: seat-balanced utility of SM-AZ against SM-AZ(-RT) against
#     w = prior_scale / T1, one line per search budget, for Dubins and SDA.
#
# Missing inputs are skipped, so this can be rerun while sweeps are in progress.

using DelimitedFiles
using LaTeXStrings
using Plots
using Printf

include(joinpath(@__DIR__, "transfer_weight_sweep_common.jl"))

const AAAI_COLUMN_WIDTH_BP = 238
const AAAI_POINTSIZE = 6
const DEPLOYED_SCALE = 2.5

function option(args, name, default)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    return args[idx + 1]
end

function read_table(path)
    data, header = readdlm(path, ','; header=true)
    col = Dict(name => j for (j, name) in enumerate(vec(header)))
    return data, col
end

function plot_tabular(dir, output; variant="full")
    path = joinpath(dir, "summary.csv")
    isfile(path) || (println("skip tabular: missing $(path)"); return nothing)
    data, col = read_table(path)
    rows = [r for r in axes(data, 1)
            if data[r, col["variant"]] == variant && iszero(Float64(data[r, col["leaf_error"]]))]
    isempty(rows) && (println("skip tabular: no $(variant) rows"); return nothing)
    budget = minimum(Int(data[r, col["target_iterations"]]) for r in rows)
    rows = filter(r -> Int(data[r, col["target_iterations"]]) == budget, rows)
    errors = sort!(unique(Float64(data[r, col["transfer_error"]]) for r in rows))
    palette = cgrad(:viridis, max(length(errors), 2); categorical=true)
    p = plot(;
        xscale=:log2, xlabel=L"\mathrm{Transfer\ weight}\ w",
        ylabel="Nash-gap reduction", legend=:outertop, legend_column=-1,
        size=(AAAI_COLUMN_WIDTH_BP, 200), titlefontsize=AAAI_POINTSIZE,
        guidefontsize=AAAI_POINTSIZE, tickfontsize=AAAI_POINTSIZE,
        legendfontsize=AAAI_POINTSIZE - 1, legendtitlefontsize=AAAI_POINTSIZE - 1,
        framestyle=:box, grid=false, fontfamily="Computer Modern",
        title=@sprintf(
            "(a) Tabular T=%d, %s; lines δR=δσ",
            budget, replace(variant, '_' => '-'),
        ),
    )
    hline!(p, [0.0]; color=:gray, linestyle=:dot, label="")
    for (k, e) in enumerate(errors)
        r = sort(filter(i -> Float64(data[i, col["transfer_error"]]) == e, rows);
                 by=i -> Float64(data[i, col["weight"]]))
        w = Float64.(data[r, col["weight"]])
        g = Float64.(data[r, col["improvement_mean"]])
        se = Float64.(data[r, col["improvement_sem"]])
        plot!(p, w, g; ribbon=1.96 .* se, fillalpha=0.15, marker=:circle, markersize=2,
              color=palette[k], label=@sprintf("%g", e))
        threshold = Float64(data[first(r), col["r5_threshold_root_range"]])
        minimum(w) <= threshold <= maximum(w) &&
            vline!(p, [threshold]; color=palette[k], linestyle=:dash, alpha=0.6, label="")
    end
    for ext in ("pdf", "png")
        savefig(p, joinpath(output, "tabular_transfer_weight_$(variant).$(ext)"))
    end
    return p
end

function learned_panel(dir, title)
    summary = summarize_sweep_dir(dir)
    isnothing(summary) && (println("skip $(title): no complete cells in $(dir)"); return nothing)
    data, col = read_table(summary)
    budgets = sort!(unique(Int.(data[:, col["queries"]])))
    palette = cgrad(:viridis, max(length(budgets), 2); categorical=true)
    p = plot(;
        xscale=:log10, xlabel=L"\mathrm{Transfer\ weight}\ w = \mathrm{prior\_scale}/T_1",
        ylabel="Seat-balanced utility vs. -RT", legend=:outertop, legend_column=-1,
        legendtitle="Queries",
        title, titlefontsize=AAAI_POINTSIZE, guidefontsize=AAAI_POINTSIZE,
        tickfontsize=AAAI_POINTSIZE, legendfontsize=AAAI_POINTSIZE - 1,
        legendtitlefontsize=AAAI_POINTSIZE - 1, framestyle=:box, grid=false,
        fontfamily="Computer Modern",
    )
    hline!(p, [0.0]; color=:gray, linestyle=:dot, label="")
    source_queries = nothing
    for (k, q) in enumerate(budgets)
        # w = 0 has no place on a log axis; it is SM-AZ(-RT) itself.
        r = sort([i for i in axes(data, 1)
                  if Int(data[i, col["queries"]]) == q && Float64(data[i, col["transfer_weight"]]) > 0];
                 by=i -> Float64(data[i, col["transfer_weight"]]))
        isempty(r) && continue
        w = Float64.(data[r, col["transfer_weight"]])
        source_queries = Float64(data[first(r), col["prior_scale"]]) / first(w)
        u = Float64.(data[r, col["seat_balanced_utility"]])
        se = Float64.(data[r, col["seat_balanced_stderr"]])
        plot!(p, w, u; ribbon=1.96 .* se, fillalpha=0.15, marker=:circle, markersize=2,
              color=palette[k], label=string(q))
    end
    isnothing(source_queries) ||
        vline!(p, [DEPLOYED_SCALE / source_queries]; color=:black, linestyle=:dash,
               alpha=0.5, label="deployed")
    return p
end

function plot_learned(dubin_dir, sda_dir, output)
    panels = filter(!isnothing, [
        learned_panel(dubin_dir, "(b) Dubins"),
        learned_panel(sda_dir, "(c) SDA"),
    ])
    isempty(panels) && return nothing
    p = plot(panels...; layout=(length(panels), 1),
             size=(AAAI_COLUMN_WIDTH_BP, 170 * length(panels)))
    for ext in ("pdf", "png")
        savefig(p, joinpath(output, "learned_transfer_weight.$(ext)"))
    end
    return p
end

function main(args=ARGS)
    root = @__DIR__
    tabular = option(args, "--tabular",
        joinpath(root, "dubin", "dubin-2026-07-19", "tabular_transfer_weight_results"))
    dubin = option(args, "--dubin",
        joinpath(root, "dubin", "dubin-2026-07-20", "transfer_weight_sweep"))
    sda = option(args, "--sda",
        joinpath(root, "sda", "sda-2026-07-21", "transfer_weight_sweep"))
    output = option(args, "--output", joinpath(root, "transfer_weight_figures"))
    mkpath(output)
    default(fontfamily="Computer Modern")
    plot_tabular(tabular, output; variant="full")
    plot_tabular(tabular, output; variant="regret_only")
    plot_learned(dubin, sda, output)
    println("Wrote figures to $(output)")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
