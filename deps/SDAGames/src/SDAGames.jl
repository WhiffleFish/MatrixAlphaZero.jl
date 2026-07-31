module SDAGames

using SatelliteDynamics
using StaticArrays
using LinearAlgebra

include("common.jl")
export FLOAT_RANGE, Observer, StateVec, idx1t3, idx4t6

include("BeliefGame/beliefgame.jl")

include("SNRGame/snrgame.jl")

include("POMDP/SDAPOMDPs.jl")

end # module SDAGames
