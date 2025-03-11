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

include("matrix.jl")

include("buffer.jl")

include("callbacks.jl")

include("solver.jl")
export AlphaZeroSolver, MCTSParams

include("tree.jl")

include("mcts.jl")

include("nn.jl")

include("train.jl")

end # module MatrixAlphaZero
