module MatrixAlphaZero

using MarkovGames
using Flux
using TensorGames
using ProtoStructs
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

include("matrix.jl")

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

end # module MatrixAlphaZero
