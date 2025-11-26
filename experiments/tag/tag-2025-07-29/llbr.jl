begin
    Pkg.activate("experiments")
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using SDAGames.SNRGame
    using SDAGames.SatelliteDynamics
    using Flux
    using Distributions
    using POMDPTools
    using POMDPs
    using ProgressMeter
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end
