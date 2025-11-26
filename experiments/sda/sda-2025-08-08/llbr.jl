using Pkg
Pkg.activate("experiments")
# using Distributed
# p = addprocs(3; exeflags="--project=$(Base.active_project())")

begin
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

# s0 = rand(initialstate(game))
# s0 = JointDubinState([3,3,0], [6,6,π])

function search_eval(planner, params::AZ.MCTSParams, game::MG, s; temperature=1.0, every=10)
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


begin
    game = SNRSDAGame(
        observer=d_observer, target=d_target, altitude_bounds=(100e3, 2e7),
    )
    oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
    oracle0 = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
    planner = AlphaZeroPlanner(game, oracle, max_iter=20, c=10.0)
    _params = AZ.MCTSParams(planner)
    s0 = rand(initialstate(game))
end

# iter, brvs1, brvs2 = search_eval(planner, params, game, s0)

res = @showprogress pmap(1:20) do i
    search_eval(planner, _params, game, s0)
end



iter = first(first(res))
brvs1 = mapreduce(hcat, res) do res_i
    res_i[2]
end
brvs2 = mapreduce(hcat, res) do res_i
    res_i[3]
end
brv_dir = mkdir(joinpath(@__DIR__, "brv"))

using DelimitedFiles
writedlm(joinpath(brv_dir, "iter.csv"), iter, ',')
writedlm(joinpath(brv_dir, "brv1.csv"), brvs1, ',')
writedlm(joinpath(brv_dir, "brv2.csv"), brvs2, ',')


plot(iter, brvs2)



plot(
    plot(iter, brvs1),
    plot(iter, brvs2),
    layout = (2,1)
)


brvs = @showprogress map(eachindex(iters)) do i
    _trees = jldopen(joinpath(treedir, "trees" * AZ.iter2string(i, 3)))["trees"]
    map(_trees) do tree
        π1 = policy1_from_tree(game, planner, tree)
        π1 = policy1_from_tree(game, planner, tree)
        Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
    end
end


##

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
# s0 = rand(initialstate(game))

hist = simulate(HistoryRecorder(max_steps=20), game, planner, s0)
lim = 2e7



plot(game, hist[1], xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500))

anim = @animate for h_i ∈ hist
    plot(game, h_i, xlims=(-lim,lim), ylims=(-lim,lim), aspect_ratio=1.0, size=(500,500))
end

gif(anim, @figdir("traj.gif"), fps=5)



br_vals = @showprogress map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)  # or your own policy function
    π2 = Tools.policy2_from_oracle(planner.oracle)  # or your own policy function
    # Tools.approx_br_value(game, oracle0, π1, s0; max_depth=5)
    brv1, brv2 = Tools.approx_br_values_both_st(game, oracle0, π1, π2, s0)
end

brvs1, brvs2 = getindex.(br_vals, 1), getindex.(br_vals, 2)
plot(
    plot(brvs1, lw=5), # Player-1 value when Player-2 plays BR to π₁ (min over a₂, expect over π₁)
    plot(-brvs2, lw=5) # Player-2 value when Player-1 plays BR to π₂ (max over a₁, expect over π₂)
)


plot(AZ.state_value(oracle, game, s0) .- brvs1, lw=5, xlabel="Training Iteration", ylabel="Observer Exploitability")
savefig(@figdir("observer-exploitability.pdf"))


plot(brvs1 .- brvs2, lw=5)


plot(br_vals, xlabel="Training Iteration", ylabel="BRV", lw=2, title="SDA SNR Game")

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

stuff = @showprogress map(1:100) do i
    search_eval(planner, MCTSParams(planner),game, s0)
end


plot(_iter, _brvs)
