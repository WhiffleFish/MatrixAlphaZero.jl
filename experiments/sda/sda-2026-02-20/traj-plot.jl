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
)
oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)

s0 = SNRGame.SDAState(
    sOSCtoCART([
        R_EARTH .+ 3e6,
        0.0,
        0.0,
        0.0,
        0.0,
        deg2rad(160)
    ]),
    sOSCtoCART([
        R_EARTH .+ 5e6,
        0.0,
        0.0,
        0.0,
        0.0,
        deg2rad(160)
    ]),
    game.epc0,
    false
)

hist = simulate(HistoryRecorder(max_steps=20), game, planner, s0)

lim = 2e7
anim = @animate for h_i ∈ hist
    plot(game, h_i, xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500))
end

gif(anim, @figdir("traj.gif"), fps=5)

hist = simulate(HistoryRecorder(max_steps=20), game, planner, s0)
begin
    p = plot(xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500), right_margin=5Plots.mm)
    for i ∈ eachindex(hist)
        if isone(i)
            plot!(p, game, hist[i])
        else
            plot!(p, game, hist[i], labels=nothing)
        end
    end
    p
end
savefig(p, @figdir("traj.pdf"))
savefig(p, @figdir("traj.svg"))

normalize(SatelliteDynamics.sun_position(s0.epc), 2)
