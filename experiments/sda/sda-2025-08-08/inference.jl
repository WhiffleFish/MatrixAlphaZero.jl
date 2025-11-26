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

oracle_info = ExperimentTools.OracleInfo(game, oracle, d_observer; n=50_000)

anim = @animate for i ∈ eachindex(oracle_info)
    plot(oracle_info[i], ms=10, clims=oracle_info.valrange)
end

gif(anim, @figdir("value.gif"), fps=5)


hist = simulate(HistoryRecorder(max_steps=20), game, planner)

lim = 2e7
plot(game, hist[3], xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500))

anim = @animate for h_i ∈ hist
    plot(game, h_i, xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500))
end

gif(anim, @figdir("traj.gif"), fps=5)

map(hist) do h_i
    entropy(h_i[:behavior][1].probs)
end |> plot

plot(hist[:r] |> collect)


anim = @animate for hi ∈ hist
    plot(game, hi, xlims=(-2e7, 2e7), ylims=(-2e7, 2e7))
end
gif(anim, @figdir("sim.gif"), fps=3)


s0 = rand(initialstate(game))
b, info = behavior_info(planner, s0)


plot(game, hist[1])

info = jldopen(joinpath(@__DIR__,"train_info.jld2"))
keys(info)
plot(reduce(vcat,info["train_losses"]))
plot(reduce(vcat,info["value_losses"]))
plot(reduce(vcat,info["policy_losses"]))

hist[1].behavior
buffer = info["buffer"]
histogram(filter(≥(50),buffer.v))
filter(≤(-50),buffer.v)
length(buffer.v)

extrema(buffer.v)

s_vec = buffer.s[argmax(buffer.v)]

s = SNRGame.SDAState(
    [s_vec[1:3] .* 1e7; s_vec[4:6] .* 1e3],
    [s_vec[7:9] .* 1e7; s_vec[10:12] .* 1e3],
    game.epc0, false
)

plot(game, s)

s.observer - s.target

game2 = SNRSDAGame(
    observer_properties = SNRGame.ObserverProperties(
            0.2,      # aperture diameter: 20 cm
            1.4,      # f-number
            9.7e-6,   # pixel size: 9.7 μm
            0.6,      # quantum efficiency
            0.5,      # dark current: 0.5 e-/pixel/s
            10.0,     # read noise: 10 e-
            2.0,      # gain: 2 e-/ADU
            0.9       # optical transmittance
        ),
        target_properties = SNRGame.TargetProperties(
            0.1,      # diameter: 1 meter
            0.175,    # albedo (from paper)
            0.5       # specular fraction: 50% specular, 50% diffuse
        ),
        conditions = SNRGame.ObservationConditions(
            1.0,      # integration time: 1 second
            4.0,      # algorithm required SNR: 4
            1,        # binning factor: 1
            30.0,     # space background: 30 mag/arcsec² (from paper's RECONSO example)
            100       # number of background pixels: 100
        )
)


reward(game2, s, (1,1))
