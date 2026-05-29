module MatrixAlphaZero

using MarkovGames
using Flux
using ProgressMeter
using LinearAlgebra
using Random
using Distributions
using JLD2
using Distributed
using POMDPs
using POMDPTools
using SpecialFunctions
using D3Trees

include("metrics.jl")
export selfplay_metrics, training_metrics, batch_metrics, oracle_metrics

include("callbacks.jl")
export MetricsCallback

include("nn.jl")
export FittedRegretModel, MultiActor, StaticFittedRegretModel

include("search/search.jl")

include("solver.jl")
export AlphaZeroSolver, SMOOSParams, AlphaZeroPlanner

include("train.jl")
export @modeldir, @model

include("vis.jl")

end # module MatrixAlphaZero
