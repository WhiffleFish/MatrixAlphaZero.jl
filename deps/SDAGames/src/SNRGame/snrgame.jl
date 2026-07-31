module SNRGame

using StaticArrays
using PDMats
using SatelliteDynamics
using RecipesBase
using Distributions
using MarkovGames
using ..SDAGames
using POMDPTools
using LinearAlgebra
using RecipesBase

include("snr.jl")
using .PhotometricSNR

include("observability.jl")

include("sda-game.jl")
export SNRSDAGame, SDAHistory

include("sda-game-simple.jl")
export SNRGameSimple

include("vis.jl")

end
