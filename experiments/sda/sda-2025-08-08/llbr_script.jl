using Pkg
# Pkg.activate("experiments")
using Distributed
p = addprocs(3; exeflags="--project=$(Base.active_project())")

@everywhere begin
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

@everywhere d_observer = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

@everywhere d_target = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

# s0 = JointDubinState([3,3,0], [6,6,π])

@everywhere function search_eval(planner, params::AZ.MCTSParams, game::MG, s; temperature=1.0, every=10)
    tree = AZ.Tree(game, s)
    (;tree_queries) = params
    γ = discount(game)
    iter = Int[]
    brvs1 = Float64[]
    brvs2 = Float64[]
    @showprogress for i ∈ 1:tree_queries
        AZ.simulate(params, tree, game, 1; temperature)
        if iszero(mod(i, every))
            π1 = Tools.policy1_from_tree(game, planner, tree)
            π2 = Tools.policy2_from_tree(game, planner, tree)
            brv1, brv2 = Tools.approx_br_values_both_st(game, oracle, π1, π2, s)
            push!(iter, i)
            push!(brvs1, brv1)
            push!(brvs2, brv2)
        end
    end
    return iter, brvs1, brvs2
end


@everywhere begin
    game = SNRSDAGame(
        observer=d_observer, target=d_target, altitude_bounds=(100e3, 2e7),
    )
    s0 = rand(initialstate(game))
    oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
    oracle0 = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
    planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
    _params = AZ.MCTSParams(planner)
end

# iter, brvs1, brvs2 = search_eval(planner, params, game, s0)

res = @showprogress pmap(1:100) do i
    search_eval(planner, _params, game, s0; every=50)
end

iter = first(first(res))
brvs1 = mapreduce(hcat, res) do res_i
    res_i[2]
end
brvs2 = mapreduce(hcat, res) do res_i
    res_i[3]
end
brv_dir = mkdir(joinpath(@__DIR__, "brv"))
joinpath(@__DIR__, "brv")

using DelimitedFiles
writedlm(joinpath(brv_dir, "iter.csv"), iter, ',')
writedlm(joinpath(brv_dir, "brv1.csv"), brvs1, ',')
writedlm(joinpath(brv_dir, "brv2.csv"), brvs2, ',')
