using Pkg
Pkg.activate("experiments")

using DelimitedFiles
using Plots
using Statistics

default(grid = false, framestyle = :box, fontfamily = "Computer Modern", label = "")

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "search_solver_diagnostics_results")

const STYLE_INFO = (
    ("regret_matching", "Regret Matching", 2),
    ("smoos", "SMOOS", 3),
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
            key_value = split(arg[3:end], "="; limit = 2)
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

function load_summary(style_dir::String)
    raw = readdlm(joinpath(style_dir, "summary.csv"), ',', String)
    header = vec(raw[1, :])
    rows = raw[2:end, :]
    col(name) = rows[:, findfirst(==(name), header)]
    num(name) = parse.(Float64, col(name))
    int(name) = parse.(Int, col(name))
    return (;
        style = col("style"),
        trial = int("trial"),
        n_states = int("n_states"),
        n_expanded = int("n_expanded"),
        root_visits = int("root_visits"),
        search_value = num("search_value"),
        policy_value = num("policy_value"),
        p1_value_vs_p2_br = num("p1_value_vs_p2_br"),
        p2_value_vs_p1_br = num("p2_value_vs_p1_br"),
        exploitability_gap = num("exploitability_gap"),
        root_regret_l2 = num("root_regret_l2"),
        root_strategy_entropy = num("root_strategy_entropy"),
        elapsed_seconds = num("elapsed_seconds"),
    )
end

function load_matrix(style_dir::String, filename::String)
    raw = readdlm(joinpath(style_dir, filename), ',', String)
    isempty(raw) && return zeros(0, 0)
    return parse.(Float64, raw[2:end, :])
end

function load_style(results_dir::String, style_name::String)
    style_dir = joinpath(results_dir, style_name)
    return (;
        summary = load_summary(style_dir),
        root_regret_p1 = load_matrix(style_dir, "root_regret_p1.csv"),
        root_regret_p2 = load_matrix(style_dir, "root_regret_p2.csv"),
        root_raw_regret_p1 = load_matrix(style_dir, "root_raw_regret_p1.csv"),
        root_raw_regret_p2 = load_matrix(style_dir, "root_raw_regret_p2.csv"),
        root_strategy_p1 = load_matrix(style_dir, "root_strategy_p1.csv"),
        root_strategy_p2 = load_matrix(style_dir, "root_strategy_p2.csv"),
    )
end

function available_styles(results_dir::String)
    filter(STYLE_INFO) do (style_name, _, _)
        isfile(joinpath(results_dir, style_name, "summary.csv"))
    end
end

function save_both(fig, output_dir::String, name::String)
    mkpath(output_dir)
    savefig(fig, joinpath(output_dir, "$(name).png"))
    savefig(fig, joinpath(output_dir, "$(name).pdf"))
end

function histogram_panel(metric_name::String, ylabel::String, data, styles)
    p = plot(; xlabel = ylabel, ylabel = "Trials", legend = :topright)
    for (style_name, style_label, color) in styles
        values = getproperty(data[style_name].summary, Symbol(metric_name))
        histogram!(p, values; bins = min(12, max(3, length(values) ÷ 2)), alpha = 0.45, c = color, label = style_label)
    end
    return p
end

function summary_figure(data, styles)
    p_exp = histogram_panel("exploitability_gap", "Exploitability Gap", data, styles)
    p_val = histogram_panel("policy_value", "Root Policy Value", data, styles)
    p_reg = histogram_panel("root_regret_l2", "Root Positive Regret L2", data, styles)
    p_ent = histogram_panel("root_strategy_entropy", "Root Strategy Entropy", data, styles)
    return plot(
        p_exp,
        p_val,
        p_reg,
        p_ent;
        layout = (2, 2),
        size = (1100, 800),
        suptitle = "Dubin Fixed-State Search Diagnostics",
    )
end

function grouped_mean_panel(data, styles, names, getters; title::String, ylabel::String, ylims = :auto)
    n = length(names)
    xbase = collect(1:n)
    width = 0.28
    p = plot(;
        title,
        xticks = (xbase, names),
        xrotation = 30,
        ylabel,
        legend = :topright,
        bottom_margin = 8 * Plots.mm,
        left_margin = 5 * Plots.mm,
        ylims,
    )
    for (k, (style_name, style_label, color)) in enumerate(styles)
        offsets = xbase .+ (k - (length(styles) + 1) / 2) * width
        vals = [mean(getter(data[style_name])) for getter in getters]
        errs = [length(getter(data[style_name])) > 1 ? std(getter(data[style_name])) : 0.0 for getter in getters]
        bar!(p, offsets, vals; yerror = errs, bar_width = width, c = color, alpha = 0.75, label = style_label)
    end
    return p
end

function metric_means_figure(data, styles)
    quality = grouped_mean_panel(
        data,
        styles,
        ["Exploitability", "Policy Value", "Search Value", "Regret L2", "Entropy"],
        (
            d -> d.summary.exploitability_gap,
            d -> d.summary.policy_value,
            d -> d.summary.search_value,
            d -> d.summary.root_regret_l2,
            d -> d.summary.root_strategy_entropy,
        );
        title = "Search Metrics",
        ylabel = "Mean",
    )
    tree = grouped_mean_panel(
        data,
        styles,
        ["States", "Expanded"],
        (
            d -> Float64.(d.summary.n_states),
            d -> Float64.(d.summary.n_expanded),
        );
        title = "Tree Size",
        ylabel = "Mean count",
        ylims = (0, :auto),
    )
    time = grouped_mean_panel(
        data,
        styles,
        ["Seconds"],
        (d -> d.summary.elapsed_seconds,);
        title = "Runtime",
        ylabel = "Mean seconds",
        ylims = (0, :auto),
    )
    return plot(quality, tree, time; layout = (1, 3), size = (1300, 460), suptitle = "Mean Diagnostics Across Trials")
end

function action_stats(mat)
    μ = vec(mean(mat; dims = 1))
    σ = size(mat, 1) > 1 ? vec(std(mat; dims = 1)) : zeros(size(mat, 2))
    return μ, σ
end

function action_panel(data, styles, field::Symbol, title::String, ylabel::String; ylims = :auto)
    n_actions = maximum(size(getproperty(data[style_name], field), 2) for (style_name, _, _) in styles)
    xbase = collect(1:n_actions)
    width = 0.28
    p = plot(; title, xlabel = "Action", ylabel, xticks = xbase, legend = :topright, ylims)
    for (k, (style_name, style_label, color)) in enumerate(styles)
        mat = getproperty(data[style_name], field)
        μ, σ = action_stats(mat)
        offsets = collect(1:length(μ)) .+ (k - (length(styles) + 1) / 2) * width
        bar!(p, offsets, μ; yerror = σ, bar_width = width, c = color, alpha = 0.75, label = style_label)
    end
    return p
end

function finite_values(values)
    return filter(isfinite, Float64.(vec(values)))
end

function kde_curve(values; points::Int = 128)
    vals = finite_values(values)
    isempty(vals) && return Float64[], Float64[]
    vmin, vmax = extrema(vals)
    scale = max(vmax - vmin, std(vals), max(abs(vmin), abs(vmax), 1.0) * 0.05)
    bandwidth = length(vals) > 1 ? 1.06 * std(vals) * length(vals)^(-1 / 5) : 0.0
    bandwidth = max(bandwidth, scale * 0.04, eps(Float64))
    lo = vmin - 2 * bandwidth
    hi = vmax + 2 * bandwidth
    grid = collect(range(lo, hi; length = points))
    norm = inv(bandwidth * sqrt(2π) * length(vals))
    density = map(grid) do y
        norm * sum(exp.(-0.5 .* ((y .- vals) ./ bandwidth) .^ 2))
    end
    max_density = maximum(density)
    if max_density > 0
        density ./= max_density
    end
    return grid, density
end

function deterministic_jitter(n::Int; width::Float64 = 0.055)
    n <= 0 && return Float64[]
    return [width * sin(12.9898 * i) for i in 1:n]
end

function violin_with_points!(p, x0::Float64, values; color, label = "", width::Float64 = 0.18)
    vals = finite_values(values)
    isempty(vals) && return p
    grid, density = kde_curve(vals)
    if !isempty(grid)
        xs = vcat(x0 .- width .* density, reverse(x0 .+ width .* density))
        ys = vcat(grid, reverse(grid))
        plot!(p, Shape(xs, ys); c = color, alpha = 0.22, linecolor = color, label)
    end
    scatter!(
        p,
        fill(x0, length(vals)) .+ deterministic_jitter(length(vals); width = 0.05),
        vals;
        markerstrokewidth = 0,
        markersize = 2.8,
        alpha = 0.55,
        c = color,
        label = "",
    )
    return p
end

function action_violin_panel(data, styles, field::Symbol, title::String, ylabel::String; ylims = :auto)
    n_actions = maximum(size(getproperty(data[style_name], field), 2) for (style_name, _, _) in styles)
    xbase = collect(1:n_actions)
    offsets = length(styles) == 1 ? [0.0] : collect(range(-0.18, 0.18; length = length(styles)))
    p = plot(;
        title,
        xlabel = "Action",
        ylabel,
        xticks = (xbase, string.(xbase)),
        legend = :topright,
        bottom_margin = 7 * Plots.mm,
        left_margin = 7 * Plots.mm,
        ylims,
    )
    for (k, (style_name, style_label, color)) in enumerate(styles)
        mat = getproperty(data[style_name], field)
        for action_idx in axes(mat, 2)
            label = action_idx == first(axes(mat, 2)) ? style_label : ""
            violin_with_points!(p, action_idx + offsets[k], mat[:, action_idx]; color, label)
        end
    end
    return p
end

function strategy_figure(data, styles)
    p1 = action_panel(data, styles, :root_strategy_p1, "Player 1", "Probability"; ylims = (0, 1))
    p2 = action_panel(data, styles, :root_strategy_p2, "Player 2", "Probability"; ylims = (0, 1))
    return plot(
        p1,
        p2;
        layout = (1, 2),
        size = (1200, 500),
        suptitle = "Root Behavioral Strategies",
        bottom_margin = 8 * Plots.mm,
        left_margin = 7 * Plots.mm,
    )
end

function strategy_violin_figure(data, styles)
    p1 = action_violin_panel(data, styles, :root_strategy_p1, "Player 1", "Probability"; ylims = (0, 1))
    p2 = action_violin_panel(data, styles, :root_strategy_p2, "Player 2", "Probability"; ylims = (0, 1))
    return plot(
        p1,
        p2;
        layout = (1, 2),
        size = (1200, 520),
        suptitle = "Root Behavioral Strategy Distributions",
        bottom_margin = 8 * Plots.mm,
        left_margin = 7 * Plots.mm,
    )
end

function regret_figure(data, styles)
    p1 = action_panel(data, styles, :root_regret_p1, "Player 1 Normalized", "Root Regret")
    p2 = action_panel(data, styles, :root_regret_p2, "Player 2 Normalized", "Root Regret")
    r1 = action_panel(data, styles, :root_raw_regret_p1, "Player 1 Raw", "Raw Root Regret")
    r2 = action_panel(data, styles, :root_raw_regret_p2, "Player 2 Raw", "Raw Root Regret")
    return plot(p1, p2, r1, r2; layout = (2, 2), size = (1100, 800), suptitle = "Root Regrets")
end

function regret_violin_figure(data, styles)
    p1 = action_violin_panel(data, styles, :root_regret_p1, "Player 1 Normalized", "Root Regret")
    p2 = action_violin_panel(data, styles, :root_regret_p2, "Player 2 Normalized", "Root Regret")
    r1 = action_violin_panel(data, styles, :root_raw_regret_p1, "Player 1 Raw", "Raw Root Regret")
    r2 = action_violin_panel(data, styles, :root_raw_regret_p2, "Player 2 Raw", "Raw Root Regret")
    return plot(
        p1,
        p2,
        r1,
        r2;
        layout = (2, 2),
        size = (1200, 850),
        suptitle = "Root Regret Distributions",
        bottom_margin = 8 * Plots.mm,
        left_margin = 7 * Plots.mm,
    )
end

function main()
    cfg = parse_cli(ARGS)
    results_dir = cfg["input"]
    output_dir = cfg["output"]
    styles = available_styles(results_dir)
    isempty(styles) && error("No diagnostic summary files found under $(results_dir)")
    data = Dict(style_name => load_style(results_dir, style_name) for (style_name, _, _) in styles)

    save_both(summary_figure(data, styles), output_dir, "summary_distributions")
    save_both(metric_means_figure(data, styles), output_dir, "summary_means")
    save_both(strategy_figure(data, styles), output_dir, "root_strategies")
    save_both(strategy_violin_figure(data, styles), output_dir, "root_strategy_violins")
    save_both(regret_figure(data, styles), output_dir, "root_regrets")
    save_both(regret_violin_figure(data, styles), output_dir, "root_regret_violins")

    println("Wrote figures to $(output_dir)")
end

main()
