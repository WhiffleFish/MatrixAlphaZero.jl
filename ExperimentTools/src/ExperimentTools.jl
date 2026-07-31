module ExperimentTools

using ArgParse
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

const AZ = MatrixAlphaZero

include("matchups.jl")
export JointPolicy, SinglePlayerAlphaZeroPolicy
export StdErrResult, ZeroSearchOracle, evaluate_joint_policy

include("dubin.jl")

include("argparse.jl")

include("sda.jl")
export SDAOutcome

end
