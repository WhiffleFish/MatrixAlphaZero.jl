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
p = plot(oracle_info[40], clims=oracle_info.valrange, title="SNR Game Observer Value Map", colorbar=nothing)
sz = 1000
plot(
    plot(oracle_info[1], ms =10, clims=oracle_info.valrange, colorbar=nothing),
    plot(oracle_info[20], ms =10, clims=oracle_info.valrange, colorbar=nothing, yticks=nothing, legend=nothing),
    plot(oracle_info[40], ms =10, clims=oracle_info.valrange, yticks=nothing, colorbar=nothing, legend=nothing),
    scatter([0,0], [0,1], zcolor=[0,3], clims=oracle_info.valrange,
        xlims=(1,1.1), xshowaxis=false, yshowaxis=false, label="", c=:magma, grid=false
    ),
    legendfontsize = 15,
    layout = @layout([grid(1, 3) a{0.07w}]),
    size=(sz*3 + 800 * 0.05 , sz+50),
    bottom_margin = 10Plots.mm,
    top_margin = 10Plots.mm,
    tickfontsize=15,
    suptitle = "Observer Game Value",
    plot_titlefont = 40,
    # arrow = Plots.arrow(:open, :head, 0.01, 0.01)
)

savefig(@figdir("value_progression.pdf"))
savefig(@figdir("value_progression-1000.png"))
