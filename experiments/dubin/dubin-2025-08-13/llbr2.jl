using Pkg
Pkg.activate("experiments")
using Distributed
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
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

@everywhere begin
    game = DubinMG()
    oracle0 = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0000.jld2"))
    oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
    planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
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
    search_eval(planner, _params, game, s0; every=50)
end

iter = first(first(res))
brvs1 = mapreduce(hcat, res) do res_i
    res_i[2]
end
brvs2 = mapreduce(hcat, res) do res_i
    res_i[3]
end

μ1 = mean(brvs1, dims=2)
σ1 = std(brvs1, dims=2)
μ2 = mean(brvs2, dims=2)
σ2 = std(brvs2, dims=2)
plot(iter, μ1, ribbon=(σ1, σ1))
plot(iter, μ2, ribbon=(σ2, σ2))



plot(iter, -μ1 .+ μ2, ribbon=dropdims(sqrt.(σ1 .^2 .+ σ2 .^ 2), dims=2), lw=2)


plot(iter, brvs1, c=2, lw=2, alpha=0.1)
plot!(iter, μ1, ribbon=σ1, lw=2, c=1)

plot(iter, -brvs2, c=2, lw=2, alpha=0.1)
plot!(iter, -μ2, ribbon=σ1, lw=2, c=1)




##

info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
keys(info)
plot(reduce(vcat, info["train_losses"]))
plot(reduce(vcat, info["value_losses"]))
plot(reduce(vcat, info["policy_losses"]))

##

res = search_eval(planner, _params, game, s0; every=50)
iter = first(res)
brvs1 = res[2]
brvs2 = res[3]

plot(
    plot(iter, brvs1),
    plot(iter, brvs2),
)

