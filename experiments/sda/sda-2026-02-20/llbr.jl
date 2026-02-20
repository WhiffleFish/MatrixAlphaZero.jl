begin
    using Pkg
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
oracle = @model(1)
planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
s0 = rand(initialstate(game))

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)
    π2 = Tools.policy2_from_oracle(planner.oracle)
    Tools.approx_br_values_both_st(game, planner.oracle, π1, π2, s0)
end

plot(
    plot(
        plot(getindex.(br_vals, 1), ylabel=L"U^1(\pi^1, \textbf{BR}(\pi^1))", c=1),
        plot(getindex.(br_vals, 2), xlabel="Training Iteration", ylabel=L"U^2(\textbf{BR}(\pi^2), \pi^2)", c=2),
        layout=(2,1)
    ),
    plot(-(getindex.(br_vals, 1) .+ getindex.(br_vals, 2)), xlabel="Training Iteration", ylabel="Exploitability", c=3),
    suptitle = "Dubin Policy Network Performance",
    layout = (1,2),
    lw=2
)

savefig(@figdir("sda-policy-network-performance.pdf"))
##

writedlm(joinpath(@__DIR__, "network_brvs.csv"), hcat(getindex.(br_vals, 1), getindex.(br_vals, 2)), ',')
