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

include("metrics.jl")
export selfplay_metrics, training_metrics, batch_metrics, oracle_metrics

include("callbacks.jl")
export MetricsCallback

include("nn.jl")
export FittedRegretModel, ActorCritic, CriticOnly, MultiActor, StaticFittedRegretModel, StaticActorCritic

include("search/search.jl")

include("solver.jl")
export AlphaZeroSolver, MCTSSearch, RegretMatchingSearch, RegretMatchingMethod, Vanilla, Plus, AlphaZeroPlanner

include("train.jl")
export @modeldir, @model

end # module MatrixAlphaZero
