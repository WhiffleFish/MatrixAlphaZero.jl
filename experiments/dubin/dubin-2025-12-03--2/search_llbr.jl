using Pkg
# Pkg.activate("experiments")
using Distributed
using DelimitedFiles
p = addprocs(3; exeflags="--project=$(Base.active_project())")

@everywhere begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Flux
    using Distributions
    using POMDPTools
    using POMDPs
    using POSGModels.Dubin
    using ProgressMeter
    # using Plots
    # default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

@everywhere begin
    game = DubinMG(V=(1.0, 1.0))
    oracle0 = @model(50)
    oracle = @model(50)
    planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=1.0)
    _params = AZ.MCTSParams(planner)
end


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

s0 = rand(initialstate(game))

res = @showprogress pmap(1:100) do i
    search_eval(planner, _params, game, s0; every=10)
end

rmprocs(p)

iter = first(first(res))
brvs1 = mapreduce(hcat, res) do res_i
    res_i[2]
end
brvs2 = mapreduce(hcat, res) do res_i
    res_i[3]
end

brv_dir = mkdir(joinpath(@__DIR__, "brv"))
writedlm(joinpath(brv_dir, "iter.csv"), iter, ',')
writedlm(joinpath(brv_dir, "brv1.csv"), brvs1, ',')
writedlm(joinpath(brv_dir, "brv2.csv"), brvs2, ',')


