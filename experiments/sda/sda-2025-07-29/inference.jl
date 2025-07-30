begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using SDAGames.SNRGame
    using SDAGames.SatelliteDynamics
    using LinearAlgebra
    using ExperimentTools
    using POMDPs
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

d_observer = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

d_target = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

game = SNRSDAGame(
    observer=d_observer, target=d_target, altitude_bounds=(100e3, 2e7),
    actions= ([-500.0, 0., 500.0], [-500.0, 0., 500.0]),
    dt = 100.0
)
oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
Flux.loadmodel!(planner, @modeldir("oracle0100.jld2"))

oracle_info = ExperimentTools.OracleInfo(game, oracle, d_observer; n=10_000)

info_i = oracle_info[3]
plot(oracle_info[50], ms=10, clims=info_i.valrange)

hist = simulate(HistoryRecorder(max_steps=100), game, planner)

plot(game, hist)

anim = @animate for hi ∈ hist
    plot(game, hi.s)
end
gif(anim, @figdir("sim.gif"))

plot(game, hist[1])
