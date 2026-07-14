using Pkg
Pkg.activate("experiments")

# Plot results produced by rm_plus_regret_diagnostics.jl.
#
# Example:
#   julia --project=experiments experiments/dubin/plot_rm_plus_regret_diagnostics.jl

using DelimitedFiles
using Plots
using Statistics

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "rm_plus_regret_diagnostics_results")

const STYLE_INFO = (
    ("rm", "RM", 2),
    ("rm_plus", "RM+", 3),
)

function parse_cli(args)
    cfg = Dict(
        "input" => DEFAULT_RESULTS_DIR,
        "output" => joinpath(DEFAULT_RESULTS_DIR, "figures"),
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key_value = split(arg[3:end], '='; limit=2)
            key = first(key_value)
            value = if length(key_value) == 2
                last(key_value)
            else
                i < length(args) || error("Missing value for --$(key)")
                i += 1
                args[i]
            end
            key in ("input", "output") || error("Unknown argument --$(key)")
            cfg[key] = value
        else
            error("Unexpected positional argument $(arg)")
        end
        i += 1
    end
    return cfg
end

function query_budgets(results_dir::String)
    isdir(results_dir) || error("Results directory does not exist: $(results_dir)")
    budgets = Int[]
    for entry in readdir(results_dir)
        startswith(entry, "queries_") || continue
        value = tryparse(Int, entry[length("queries_") + 1:end])
        isnothing(value) || push!(budgets, value)
    end
    sort!(unique!(budgets))
    isempty(budgets) && error("No queries_* result directories found under $(results_dir)")
    return budgets
end

function load_matrix(style_dir::String, filename::String)
    path = joinpath(style_dir, filename)
    isfile(path) || error("Missing diagnostic matrix: $(path)")
    raw = readdlm(path, ',', String)
    size(raw, 1) > 1 || return zeros(0, max(size(raw, 2), 0))
    return parse.(Float64, raw[2:end, :])
end

function load_style(results_dir::String, queries::Int, style_name::String)
    style_dir = joinpath(results_dir, "queries_$(queries)", style_name)
    return (;
        raw_p1=load_matrix(style_dir, "root_raw_regret_p1.csv"),
        raw_p2=load_matrix(style_dir, "root_raw_regret_p2.csv"),
        sqrt_p1=load_matrix(style_dir, "root_sqrt_normalized_regret_p1.csv"),
        sqrt_p2=load_matrix(style_dir, "root_sqrt_normalized_regret_p2.csv"),
        average_p1=load_matrix(style_dir, "root_average_regret_p1.csv"),
        average_p2=load_matrix(style_dir, "root_average_regret_p2.csv"),
        strategy_p1=load_matrix(style_dir, "root_strategy_p1.csv"),
        strategy_p2=load_matrix(style_dir, "root_strategy_p2.csv"),
    )
end

function available_styles(results_dir::String, queries::Int)
    styles = filter(STYLE_INFO) do (style_name, _, _)
        isfile(joinpath(results_dir, "queries_$(queries)", style_name, "root_raw_regret_p1.csv"))
    end
    isempty(styles) && error("No RM or RM+ results found for $(queries) queries")
    return styles
end

function save_both(fig, output_dir::String, name::String)
    mkpath(output_dir)
    savefig(fig, joinpath(output_dir, "$(name).png"))
    savefig(fig, joinpath(output_dir, "$(name).pdf"))
end

finite_values(values) = filter(isfinite, Float64.(vec(values)))

function kde_curve(values; points::Int=128)
    vals = finite_values(values)
    isempty(vals) && return Float64[], Float64[]
    vmin, vmax = extrema(vals)
    sample_std = length(vals) > 1 ? std(vals) : 0.0
    scale = max(vmax - vmin, sample_std, max(abs(vmin), abs(vmax), 1.0) * 0.05)
    bandwidth = length(vals) > 1 ? 1.06 * sample_std * length(vals)^(-1 / 5) : 0.0
    bandwidth = max(bandwidth, scale * 0.04, eps(Float64))
    grid = collect(range(vmin - 2 * bandwidth, vmax + 2 * bandwidth; length=points))
    norm = inv(bandwidth * sqrt(2pi) * length(vals))
    density = map(grid) do y
        norm * sum(exp.(-0.5 .* ((y .- vals) ./ bandwidth) .^ 2))
    end
    max_density = maximum(density)
    max_density > 0 && (density ./= max_density)
    return grid, density
end

function deterministic_jitter(n::Int; width::Float64=0.05)
    return [width * sin(12.9898 * i) for i in 1:n]
end

function violin_with_points!(p, x0::Float64, values; color, label="", width::Float64=0.18)
    vals = finite_values(values)
    isempty(vals) && return p
    grid, density = kde_curve(vals)
    if !isempty(grid)
        xs = vcat(x0 .- width .* density, reverse(x0 .+ width .* density))
        ys = vcat(grid, reverse(grid))
        plot!(p, Shape(xs, ys); c=color, alpha=0.22, linecolor=color, label)
    end
    scatter!(
        p,
        fill(x0, length(vals)) .+ deterministic_jitter(length(vals)),
        vals;
        markerstrokewidth=0,
        markersize=2.8,
        alpha=0.55,
        c=color,
        label="",
    )
    return p
end

function action_violin_panel(data, styles, field::Symbol, title::String, ylabel::String; ylims=:auto)
    n_actions = maximum(size(getproperty(data[style_name], field), 2) for (style_name, _, _) in styles)
    offsets = length(styles) == 1 ? [0.0] : collect(range(-0.18, 0.18; length=length(styles)))
    p = plot(;
        title,
        xlabel="Action",
        ylabel,
        xticks=(1:n_actions, string.(1:n_actions)),
        legend=:topright,
        bottom_margin=7 * Plots.mm,
        left_margin=7 * Plots.mm,
        ylims,
    )
    for (style_idx, (style_name, style_label, color)) in enumerate(styles)
        matrix = getproperty(data[style_name], field)
        for action_idx in axes(matrix, 2)
            label = action_idx == first(axes(matrix, 2)) ? style_label : ""
            violin_with_points!(p, action_idx + offsets[style_idx], matrix[:, action_idx]; color, label)
        end
    end
    hline!(p, [0.0]; c=:black, alpha=0.25, linewidth=1, label="")
    return p
end

function distribution_figure(data, styles, p1_field::Symbol, p2_field::Symbol, title::String, ylabel::String; ylims=:auto)
    p1 = action_violin_panel(data, styles, p1_field, "Player 1", ylabel; ylims)
    p2 = action_violin_panel(data, styles, p2_field, "Player 2", ylabel; ylims)
    return plot(
        p1,
        p2;
        layout=(1, 2),
        size=(1200, 520),
        suptitle=title,
        bottom_margin=8 * Plots.mm,
        left_margin=7 * Plots.mm,
    )
end

function positive_l2(row)
    return sqrt(sum(abs2, max.(row, 0.0)))
end

function negative_l2(row)
    return sqrt(sum(abs2, min.(row, 0.0)))
end

function positive_hhi(row)
    positive = max.(row, 0.0)
    total = sum(positive)
    total > 0 || return NaN
    return sum(abs2, positive ./ total)
end

positive_support(row) = count(>(0.0), row)

function row_metric(matrix, metric)
    return [metric(row) for row in eachrow(matrix)]
end

function metric_mean_std(matrix, metric)
    values = finite_values(row_metric(matrix, metric))
    isempty(values) && return NaN, NaN
    return mean(values), length(values) > 1 ? std(values) : 0.0
end

function trend_panel(all_data, budgets, player::Int, metric, title::String, ylabel::String; ylims=:auto)
    field = Symbol("sqrt_p$(player)")
    p = plot(;
        title="Player $(player): $(title)",
        xlabel="Tree queries",
        ylabel,
        xscale=:log10,
        xticks=(budgets, string.(budgets)),
        legend=:topright,
        left_margin=6 * Plots.mm,
        bottom_margin=6 * Plots.mm,
        ylims,
    )
    for (style_name, style_label, color) in STYLE_INFO
        means = Float64[]
        stds = Float64[]
        for queries in budgets
            matrix = getproperty(all_data[(queries, style_name)], field)
            mu, sigma = metric_mean_std(matrix, metric)
            push!(means, mu)
            push!(stds, sigma)
        end
        plot!(p, budgets, means; ribbon=stds, marker=:circle, c=color, alpha=0.85, label=style_label)
    end
    return p
end

function concentration_trends(all_data, budgets)
    panels = Any[]
    for player in 1:2
        push!(panels, trend_panel(all_data, budgets, player, positive_l2, "Positive L2", "L2 of [R]+ / sqrt(T)"; ylims=(0, :auto)))
        push!(panels, trend_panel(all_data, budgets, player, negative_l2, "Negative L2", "L2 of [R]- / sqrt(T)"; ylims=(0, :auto)))
        push!(panels, trend_panel(all_data, budgets, player, positive_hhi, "Positive HHI", "HHI"; ylims=(0, 1)))
        push!(panels, trend_panel(all_data, budgets, player, positive_support, "Positive Support", "Number of actions"; ylims=(0, 3)))
    end
    return plot(
        panels...;
        layout=(2, 4),
        size=(1600, 850),
        suptitle="RM vs RM+ Regret Concentration Across Search Budgets",
    )
end

function dominant_fractions(matrix)
    n_actions = size(matrix, 2)
    dominant = [argmax(row) for row in eachrow(matrix)]
    return [count(==(action), dominant) / length(dominant) for action in 1:n_actions]
end

function dominant_mode_panel(all_data, budgets, player::Int)
    field = Symbol("sqrt_p$(player)")
    n_actions = size(getproperty(all_data[(first(budgets), "rm")], field), 2)
    p = plot(;
        title="Player $(player)",
        xlabel="Tree queries",
        ylabel="Dominant-action fraction",
        xscale=:log10,
        xticks=(budgets, string.(budgets)),
        ylims=(0, 1),
        legend=:outertopright,
        left_margin=6 * Plots.mm,
        bottom_margin=6 * Plots.mm,
    )
    for (style_name, style_label, _) in STYLE_INFO
        linestyle = style_name == "rm" ? :solid : :dash
        fractions = [dominant_fractions(getproperty(all_data[(queries, style_name)], field)) for queries in budgets]
        for action in 1:n_actions
            values = [fraction[action] for fraction in fractions]
            plot!(
                p,
                budgets,
                values;
                marker=:circle,
                c=action,
                linestyle,
                label="$(style_label) action $(action)",
            )
        end
    end
    return p
end

function dominant_modes_figure(all_data, budgets)
    return plot(
        dominant_mode_panel(all_data, budgets, 1),
        dominant_mode_panel(all_data, budgets, 2);
        layout=(1, 2),
        size=(1350, 520),
        suptitle="Dominant Regret Modes Across Trials",
    )
end

function main()
    cfg = parse_cli(ARGS)
    results_dir = cfg["input"]
    output_dir = cfg["output"]
    budgets = query_budgets(results_dir)

    all_data = Dict{Tuple{Int,String},Any}()
    for queries in budgets
        styles = available_styles(results_dir, queries)
        length(styles) == length(STYLE_INFO) || error("Both RM and RM+ results are required for $(queries) queries")
        data = Dict(style_name => load_style(results_dir, queries, style_name) for (style_name, _, _) in styles)
        for (style_name, _, _) in styles
            all_data[(queries, style_name)] = data[style_name]
        end

        save_both(
            distribution_figure(data, styles, :raw_p1, :raw_p2, "Raw Cumulative Regret Sums ($(queries) Queries)", "Cumulative regret"),
            output_dir,
            "root_raw_regret_violins_$(queries)",
        )
        save_both(
            distribution_figure(data, styles, :sqrt_p1, :sqrt_p2, "Sqrt-Normalized Regrets ($(queries) Queries)", "R / sqrt(T)"),
            output_dir,
            "root_sqrt_normalized_regret_violins_$(queries)",
        )
        save_both(
            distribution_figure(data, styles, :average_p1, :average_p2, "Average Regrets ($(queries) Queries)", "R / T"),
            output_dir,
            "root_average_regret_violins_$(queries)",
        )
        save_both(
            distribution_figure(data, styles, :strategy_p1, :strategy_p2, "Root Average Strategies ($(queries) Queries)", "Probability"; ylims=(0, 1)),
            output_dir,
            "root_strategy_violins_$(queries)",
        )
    end

    save_both(concentration_trends(all_data, budgets), output_dir, "regret_concentration_trends")
    save_both(dominant_modes_figure(all_data, budgets), output_dir, "dominant_regret_modes")
    println("Wrote figures to $(output_dir)")
end

main()
