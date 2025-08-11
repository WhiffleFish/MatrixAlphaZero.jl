module MatrixAlphaZero

using MarkovGames
using Flux
using TensorGames
using ProgressMeter
using LinearAlgebra
using Random
using Distributions
using DataStructures
using JLD2
using Distributed
using POMDPs
using POMDPTools
using SpecialFunctions
using D3Trees

include("matrix.jl")
export PATHSolver, RegretSolver

include("buffer.jl")

include("callbacks.jl")

include("nn.jl")
export ActorCritic, MultiActor

include("solver.jl")
export AlphaZeroSolver, MCTSParams, AlphaZeroPlanner

include("tree.jl")

include("mcts.jl")

include("train.jl")
export @modeldir

include("vis.jl")

## Experimental
include(joinpath("bounds", "bounds.jl"))
using .Bounds


end # module MatrixAlphaZero
