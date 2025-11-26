using DelimitedFiles
using Statistics
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

brv_dir = joinpath(@__DIR__, "brv")
iter = readdlm(joinpath(brv_dir, "iter.csv"), ',')
brv1 = readdlm(joinpath(brv_dir, "brv1.csv"), ',')
brv2 = readdlm(joinpath(brv_dir, "brv2.csv"), ',')


plot_kwargs = (;lw=5, alpha=0.5)
plot(
    plot(iter, brv1; c=1, plot_kwargs...),
    plot(iter, brv2; c=2, plot_kwargs...)
)

μ1 = mean(brv1, dims=2)
μ2 = mean(brv2, dims=2)
σ1 = std(brv1, dims=2) ./ √length(iter)
σ2 = std(brv2, dims=2) ./ √length(iter)
plot(μ1, ribbon =  (σ1, σ1))
plot(μ2, ribbon =  (σ2, σ2))
plot(μ1 .- μ2, ribbon = (sqrt.(σ1 .^2 .+ σ2 .^2), sqrt.(σ1 .^2 .+ σ2 .^2)))
