begin
    Pkg.activate("experiments")
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using POSGModels.Dubin
    using Flux
    using POMDPTools
    using POMDPs
    using ProgressMeter
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

game = DubinMG()
oracle0 = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
s0 = rand(initialstate(game))
# s0 = JointDubinState([3,3,0], [6,6,π])

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)
    π2 = Tools.policy2_from_oracle(planner.oracle)
    Tools.approx_br_values_both_st(game, oracle, π1, π2, s0)
end

plot(getindex.(br_vals, 1), xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")
plot(getindex.(br_vals, 2), xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")

x,y,t = solve(planner.matrix_solver, AZ.oracle_matrix_game(game, Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0000.jld2")), s0))
x
AZ.state_policy(
    Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0000.jld2")), 
    game, 
    s0
)

function policy1_from_value_oracle(oracle, matrix_solver)
    return function (game, s)
        x,y,t = solve(matrix_solver, AZ.oracle_matrix_game(game, oracle, s))
        return x
    end
end

function policy1_from_planner(planner::AlphaZeroPlanner)
    return function (game, s)
        σ = behavior(planner, s)
        return σ[1].probs
    end
end


br_vals2 = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = policy1_from_value_oracle(planner.oracle, planner.matrix_solver)
    Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
end

plot(br_vals2, xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")


planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
br_vals3 = map(readdir(@modeldir; join=true)) do modelpath
    @show modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = policy1_from_planner(planner)
    Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
end

plot(br_vals3, xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")


##
function policy1_from_tree(game, planner, tree)
    γ = discount(game)
    (;matrix_solver) = planner
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(AZ.node_matrix_game(tree, 1.0, idx, γ))
            # @info "NOT FOUND"
            x,y = AZ.state_policy(planner.oracle, game, s)
            return x
        else
            # @info "FOUND"
            A = AZ.node_matrix_game(tree, 1.0, idx, γ)
            x,y,t = solve(matrix_solver, A)
            return x
        end
    end
end

b, info = behavior_info(planner, s0)

p1 = policy1_from_tree(game, planner, info.tree)

sp, r = @gen(:sp, :r)(game, s0, (1,3))


iters = 1:10:1010

n_mc = 100
trees = @showprogress map(iters) do iter
    map(1:n_mc) do i
        b, info = behavior_info(AlphaZeroPlanner(planner; max_iter=iter), s0)
        info.tree
    end
end

using JLD2
treedir = joinpath(@__DIR__, "trees")
mkdir(treedir)

# FIXME: append .jld2 to filepath
foreach(eachindex(trees)) do i
    path = joinpath(treedir, "trees" * AZ.iter2string(i, 3))
    jldsave(path; trees=trees[i])
end

i = 1

brvs = @showprogress map(trees) do _trees
    map(_trees) do tree
        π1 = policy1_from_tree(game, planner, tree)
        Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
    end
end

plot(iters, brvs)


brvs = @showprogress map(eachindex(iters)) do i
    _trees = jldopen(joinpath(treedir, "trees" * AZ.iter2string(i, 3)))["trees"]
    map(_trees) do tree
        π1 = policy1_from_tree(game, planner, tree)
        Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
    end
end

using Statistics
plot(iters, reduce(hcat, brvs)', lw=5, alpha=0.5)
μ, σ = map(mean, brvs), map(std, brvs)
plot(
    iters, μ, 
    fillrange=(μ .- σ, μ .+ σ), 
    fillalpha=0.25, 
    lw=5,
    fillcolor=2,
    xlabel = "Search Iterations",
    ylabel = "BRV",
    title = "Dubin Policy Robustness"
)


AZ.state_value(oracle, game, s0)
##
function search_eval(planner, params::MCTSParams, game::MG, s; temperature=1.0, every=10)
    tree = AZ.Tree(game, s)
    (;tree_queries) = params
    γ = discount(game)
    iter = Int[]
    brvs = Float64[]
    @showprogress for i ∈ 1:tree_queries
        AZ.simulate(params, tree, game, 1; temperature)
        if iszero(mod(i, every))
            π1 = policy1_from_tree(game, planner, tree)
            brv = Tools.approx_br_value(game, planner.oracle, π1, s; max_depth=5)
            push!(iter, i)
            push!(brvs, brv)
        end
    end
    return iter, brvs
end

function MCTSParams(planner::AlphaZeroPlanner; kwargs...)
    return MCTSParams(;
        tree_queries = planner.max_iter,
        c = planner.c,
        max_depth = planner.max_depth,
        max_time = planner.max_time,
        matrix_solver = planner.matrix_solver,
        oracle = planner.oracle,
        kwargs...
    )
end

_iter, _brvs = search_eval(planner, MCTSParams(planner),game, s0)

plot(_iter, _brvs)
