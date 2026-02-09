using Plots
using DelimitedFiles
using Statistics
using LaTeXStrings
using ExperimentTools
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

brv_dir = joinpath(@__DIR__, "brv")
iter = readdlm(joinpath(brv_dir, "iter.csv"), ',')
brvs1 = readdlm(joinpath(brv_dir, "brv1.csv"), ',')
brvs2 = readdlm(joinpath(brv_dir, "brv2.csv"), ',')

μ1 = mean(brvs1, dims=2)
σ1 = std(brvs1, dims=2)
μ2 = mean(brvs2, dims=2)
σ2 = std(brvs2, dims=2)

μtot = -(μ1 .+ μ2) |> vec
σtot = sqrt.(σ1 .^2 .+ σ2 .^2) |> vec

p_brv1 = plot(iter, μ1, ribbon=σ1, c=1, lw=2, xlabel="Search Iterations", ylabel=L"U^1(\pi^1, \textbf{BR}(\pi^1))")
p_brv2 = plot(iter, μ2, ribbon=σ2, c=2, lw=2, xlabel="Search Iterations", ylabel=L"U^2(\textbf{BR}(\pi^2), \pi^2)")

p_tot = plot(iter, 0.5 * μtot, ribbon=0.5 * σtot,
    xlabel = "Search Iterations",
    ylabel = "Exploitability",
    fillcolor=:lightgray,
    lw=5,
    fillalpha=0.5,
    c=3
    # size=(500, 500),
    # size=(500Plots.mm, 500Plots.mm)
)

plot(
    plot(p_brv1, p_brv2, layout=(1,2)),
    p_tot,
    layout = (2,1),
    suptitle = "Dubin Simultaneous MCTS Performance",
    size = (500, 700)
)
savefig(@figdir("simultaneous-mcts-performance.pdf"))

plot(
    plot(p_brv1, p_brv2, layout=(2,1)),
    p_tot,
    layout = (1,2),
    suptitle = "Dubin Simultaneous MCTS Performance",
    size = (900, 500),
    left_margin = 4Plots.mm,
    bottom_margin = 4Plots.mm
)
savefig(@figdir("simultaneous-mcts-performance-horizontal.pdf"))
