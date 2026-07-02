using Pkg
Pkg.activate("experiments")

using DelimitedFiles
using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random
using Plots

const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const DubinTools = ExperimentTools.Dubin
const EXPERIMENT_DIR = @__DIR__

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

data, headers = readdlm(joinpath(@__DIR__,"transfer_weight_sweep_500.csv"), ',', header=true)
vheaders = vec(headers)

ws = data[:,findfirst(==("transfer_weight"), vheaders)]

μ1 = data[:,findfirst(==("az_p1_vs_heuristic_reward"), vheaders)]
σ1 = data[:,findfirst(==("az_p1_vs_heuristic_stderr_reward"), vheaders)]

μ2 = data[:,findfirst(==("heuristic_vs_az_p2_reward"), vheaders)]
σ2 = data[:,findfirst(==("heuristic_vs_az_p2_stderr_reward"), vheaders)]

plot_err(x, y, σ; kwargs...) = plot(x, y, ribbon=(σ, σ); kwargs...)

plot_err(ws, μ1, σ1)
plot_err(ws, μ2, σ2)
