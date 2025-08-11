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
    using JLD2
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
plot(oracle_info[30], ms=10, clims=info_i.valrange)

hist = simulate(HistoryRecorder(max_steps=100), game, planner)

plot(collect(hist[:r]))
plot(AZ.batch_state_value(oracle, game, collect(hist[:s])))

plot(game, hist)

anim = @animate for hi ∈ hist
    plot(game, hi.s)
end
gif(anim, @figdir("sim.gif"))

plot(game, hist[1])

##
info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
ks = keys(info)

plot(reduce(vcat,info["train_losses"]))
plot(reduce(vcat,info["value_losses"]))
plot(reduce(vcat,info["policy_losses"]))

buffer = info["buffer"]
buffer.s[1]
V = AZ.value(oracle, reduce(hcat, buffer.s))
histogram(buffer.v, bins=20)
histogram!(V, bins=20)

ΔV = V .- buffer.v
histogram(ΔV, bins=20)

maximum(buffer.v)
buf_idx = argmax(buffer.v)
s_vec = buffer.s[buf_idx]

s = SNRGame.SDAState(
    [s_vec[1:3] .* 1e7;s_vec[4:6] .* 1e3],
    [s_vec[7:9] .* 1e7; s_vec[10:12] .* 1e3],
    game.epc0, false
)


hist = simulate(HistoryRecorder(max_steps=100), game, planner, s)
plot(collect(hist[:r]))
