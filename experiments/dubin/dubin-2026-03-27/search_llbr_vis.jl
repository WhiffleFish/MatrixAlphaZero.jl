using DelimitedFiles
using Statistics
using Plots
using LaTeXStrings

default(grid = false, framestyle = :box, fontfamily = "Computer Modern", label = "")

const EXPERIMENT_DIR = @__DIR__
const RESULTS_DIR = joinpath(EXPERIMENT_DIR, "search_llbr_results")
const STYLE_ORDER = (
    ("matrix_game", "Greedy Matrix", 1),
    ("regret_matching", "Regret Matching", 2),
    ("exp3", "Exp3", 3),
)

function load_style_result(style_name::String)
    style_dir = joinpath(RESULTS_DIR, style_name)
    iter = vec(readdlm(joinpath(style_dir, "iter.csv"), ','))
    brv1 = readdlm(joinpath(style_dir, "brv1.csv"), ',')
    brv2 = readdlm(joinpath(style_dir, "brv2.csv"), ',')
    values = readdlm(joinpath(style_dir, "value.csv"), ',')
    return (; iter, brv1, brv2, values)
end

function style_summary(style_name::String)
    data = load_style_result(style_name)
    μ1 = vec(mean(data.brv1, dims = 2))
    σ1 = vec(std(data.brv1, dims = 2))
    μ2 = vec(mean(data.brv2, dims = 2))
    σ2 = vec(std(data.brv2, dims = 2))
    μv = vec(mean(data.values, dims = 2))
    σv = vec(std(data.values, dims = 2))
    μexp = -0.5 .* (μ1 .+ μ2)
    σexp = 0.5 .* sqrt.(σ1 .^ 2 .+ σ2 .^ 2)
    return (; data.iter, μ1, σ1, μ2, σ2, μv, σv, μexp, σexp)
end

function main()
    summaries = Dict(style_name => style_summary(style_name) for (style_name, _, _) in STYLE_ORDER)

    p_brv1 = plot(
        xlabel = "Search Iterations",
        ylabel = L"U^1(\pi^1, \mathbf{BR}(\pi^1))",
        legend = :topright,
    )
    p_brv2 = plot(
        xlabel = "Search Iterations",
        ylabel = L"U^2(\mathbf{BR}(\pi^2), \pi^2)",
        legend = :topright,
    )
    p_exp = plot(
        xlabel = "Search Iterations",
        ylabel = "Exploitability",
        legend = :topright,
    )
    p_val = plot(
        xlabel = "Search Iterations",
        ylabel = "Root Value",
        legend = :topright,
    )

    for (style_name, style_label, color) in STYLE_ORDER
        summary = summaries[style_name]
        plot!(p_brv1, summary.iter, summary.μ1; ribbon = summary.σ1, lw = 2, c = color, label = style_label)
        plot!(p_brv2, summary.iter, summary.μ2; ribbon = summary.σ2, lw = 2, c = color, label = style_label)
        plot!(p_exp, summary.iter, summary.μexp; ribbon = summary.σexp, lw = 2, c = color, label = style_label)
        plot!(p_val, summary.iter, summary.μv; ribbon = summary.σv, lw = 2, c = color, label = style_label)
    end

    fig = plot(
        p_brv1,
        p_brv2,
        p_exp,
        p_val;
        layout = (2, 2),
        size = (1000, 800),
        suptitle = "Dubin Search Style Comparison",
    )

    savefig(fig, joinpath(EXPERIMENT_DIR, "dubin-search-style-comparison.pdf"))
    savefig(fig, joinpath(EXPERIMENT_DIR, "dubin-search-style-comparison.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
