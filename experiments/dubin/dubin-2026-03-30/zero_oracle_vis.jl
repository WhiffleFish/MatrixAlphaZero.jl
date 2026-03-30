using Pkg
Pkg.activate("experiments")

using DelimitedFiles
using Statistics
using Plots
using LaTeXStrings

default(grid = false, framestyle = :box, fontfamily = "Computer Modern", label = "")

const RESULTS_DIR = joinpath(@__DIR__, "zero_oracle_results")
const STYLE_ORDER = (
    ("matrix_game",     "Matrix Game",     1),
    ("regret_matching", "Regret Matching", 2),
    ("exp3",            "Exp3",            3),
)

function load_style(style_name::String)
    dir  = joinpath(RESULTS_DIR, style_name)
    iter = vec(readdlm(joinpath(dir, "iter.csv"),  ','))
    brv1 = readdlm(joinpath(dir, "brv1.csv"),      ',')
    brv2 = readdlm(joinpath(dir, "brv2.csv"),      ',')
    vals = readdlm(joinpath(dir, "value.csv"),      ',')
    return (; iter, brv1, brv2, vals)
end

function summarize(style_name::String)
    d    = load_style(style_name)
    μ1   = vec(mean(d.brv1; dims = 2))
    σ1   = vec(std(d.brv1;  dims = 2))
    μ2   = vec(mean(d.brv2; dims = 2))
    σ2   = vec(std(d.brv2;  dims = 2))
    μv   = vec(mean(d.vals; dims = 2))
    σv   = vec(std(d.vals;  dims = 2))
    # exploitability = 0.5 * (-brv2 - brv1)  [≥ 0, zero at Nash]
    μexp = -0.5 .* (μ1 .+ μ2)
    σexp = 0.5 .* sqrt.(σ1 .^ 2 .+ σ2 .^ 2)
    return (; d.iter, μ1, σ1, μ2, σ2, μv, σv, μexp, σexp)
end

summaries = Dict(name => summarize(name) for (name, _, _) in STYLE_ORDER)

p_brv1 = plot(xlabel = "Search iterations", ylabel = L"V^1(\pi^1,\,\mathbf{BR}(\pi^1))", legend = :bottomright)
p_brv2 = plot(xlabel = "Search iterations", ylabel = L"V^2(\mathbf{BR}(\pi^2),\,\pi^2)", legend = :topright)
p_exp  = plot(xlabel = "Search iterations", ylabel = "Exploitability",  legend = :topright)
p_val  = plot(xlabel = "Search iterations", ylabel = "Root value",      legend = :topright)

for (name, label, color) in STYLE_ORDER
    s = summaries[name]
    plot!(p_brv1, s.iter, s.μ1;   ribbon = s.σ1,   lw = 2, c = color, label = label)
    plot!(p_brv2, s.iter, s.μ2;   ribbon = s.σ2,   lw = 2, c = color, label = label)
    plot!(p_exp,  s.iter, s.μexp; ribbon = s.σexp, lw = 2, c = color, label = label)
    plot!(p_val,  s.iter, s.μv;   ribbon = s.σv,   lw = 2, c = color, label = label)
end

fig = plot(
    p_brv1, p_brv2, p_exp, p_val;
    layout   = (2, 2),
    size     = (1000, 800),
    suptitle = "Zero-Oracle Search Method Comparison (Dubin)",
)

savefig(fig, joinpath(@__DIR__, "zero-oracle-search-comparison.pdf"))
savefig(fig, joinpath(@__DIR__, "zero-oracle-search-comparison.png"))
println("Saved figures to $(@__DIR__)")
