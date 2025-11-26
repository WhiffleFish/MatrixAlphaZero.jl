begin
    using FictitiousPlay
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Plots
    using POSGModels.DiscreteTag
    using Flux
    using Statistics
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

game = TagMG()
sol = FictitiousPlaySolver(
    verbose     = true, 
    iter        = 100,
    threaded    = true,
)
pol = solve(sol, game)
FictitiousPlay.clean_policy!(pol)

V = FictitiousPlay.policy_value(FictitiousPlay.PolicyEvaluator(), pol, game)
P = pol.policy_mats[1]
S = states(game)
VS = mapreduce(hcat, S) do s
    MarkovGames.convert_s(Vector{Float32}, s, game)
end

oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle, max_iter=1000)
model_nums = eachindex(readdir(@modeldir)) .- 1
Flux.loadmodel!(planner, @modeldir("oracle0100.jld2"))


V_az_full = map(model_nums) do i
    Flux.loadmodel!(planner, joinpath(@modeldir, "oracle"*AZ.iter2string(i)*".jld2"))
    AZ.value(oracle, VS)
end
P_az_full = map(model_nums) do i
    Flux.loadmodel!(planner, joinpath(@modeldir, "oracle"*AZ.iter2string(i)*".jld2"))
    AZ.policy(oracle, VS)
end

pol_diffs = map(1:2) do i
    map(P_az_full) do P_az
        P_az = P_az[i]
        vec(mean(abs.(P_az .- pol.policy_mats[i]'), dims=1))
    end
end

pol_crossentropies = mapreduce(hcat, 1:2) do i
    map(P_az_full) do P_az
        Flux.crossentropy(P_az[i], pol.policy_mats[i]')
    end
end

value_mse = map(V_az_full) do V_az
    Flux.mse(V_az, V)
end

V_diffs = map(V_az_full) do V_az
    V_az .- V
end

plot(
    axes(pol_crossentropies,1) .- 1,
    pol_crossentropies,
    xlabel = "Training Iteration",
    ylabel = "Policy Crossentropy Error",
    title = "Discrete Tag AlphaZero / FP Comparison",
    labels = ["Pursuer" "Evader"],
    lw = 2
)
savefig(@figdir("AZ-FP-policy-error.pdf"))
savefig(@figdir("AZ-FP-policy-error.png"))

plot(
    eachindex(value_mse) .- 1,
    value_mse,
    xlabel = "Training Iteration",
    ylabel = "Value MSE",
    title = "Discrete Tag AlphaZero / FP Comparison",
    lw = 2
)
savefig(@figdir("AZ-FP-value-mse.pdf"))
savefig(@figdir("AZ-FP-value-mse.png"))



S = states(game)
pursuer_state = Coord(1,1)

function plot_value(game, oracle, x, y; kwargs...)
    V = zeros(game.floor...)
    for i ∈ 1:game.floor[1], j ∈ 1:game.floor[2]
        s_vec = MarkovGames.convert_s(
            Vector{Float32}, 
            TagState(Coord(x,y), Coord(i,j), false), 
            game
        )
        V[i,j] = only(AZ.value(oracle, s_vec))
    end
    return heatmap(V', aspect_ratio=1.0; kwargs...)
end

plot_value(game, oracle, 5, 5)


## anim

pol_anim = @animate for pol_diff in pol_diffs[1]
    plot(pol_diff, label="")
end

gif(pol_anim, @figdir("pol_anim.gif"), fps=3)

anim = @animate for V_az in V_az_full
    plot(V)
    plot!(V_az)
end

gif(anim, joinpath(@__DIR__, "az_value.gif"), fps=3)


anim = @animate for V_az in V_az_full
    plot(V_az .- V, label="")
end

gif(anim, joinpath(@__DIR__, "az_value_diff.gif"), fps=3)

## where do the policies really differ?
planner = AlphaZeroPlanner(game, oracle, max_iter=10, c=10.0)
Flux.loadmodel!(planner, joinpath(@modeldir, "oracle0100.jld2"))

perms = sortperm(last(pol_diffs[2]), rev=true)
perm_idx = 1
last(pol_diffs[2])[perms[perm_idx]]
s = S[perms[perm_idx]]
last(P_az_full)[2][:,perms[perm_idx]]
pol.policy_mats[2][perms[perm_idx], :]

b, info = behavior_info(planner, S[perms[perm_idx]]);
b[2]

bs = map(1:100) do i
    behavior(planner, S[perms[perm_idx]])[2].probs
end

μ = mean(reduce(hcat, bs); dims=2) |> vec
σ = std(reduce(hcat, bs); dims=2) |> vec

bar(1:4, μ, yerror=3 .* σ)


using StatsPlots

violin(
    ["up" "right" "down" "left"],
    reduce(hcat, bs)',
    title = "Action probabilities 10 mcts iter"
)

planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
Flux.loadmodel!(planner, joinpath(@modeldir, "oracle0010.jld2"))
bs = map(1:100) do i
    behavior(planner, S[perms[perm_idx]])[2].probs
end

violin(
    ["up" "right" "down" "left"],
    reduce(hcat, bs)',
    title = "Action probabilities 100 mcts iter"
)

plot(game, s, ms=10)

AZ.policy(planner.oracle, MarkovGames.convert_s(Vector{Float32}, s, game))[2]


##
planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
Flux.loadmodel!(planner, joinpath(@modeldir, "oracle0100.jld2"))

s = rand(initialstate(game))
s_idx = MarkovGames.stateindex(game, s)
V[s_idx]

σ, info = behavior_info(planner, s)
info.tree |> propertynames
mat = AZ.node_matrix_game(info.tree, 1.0, 1, discount(game))
x,y,v = solve(AZ.PATHSolver(), mat)

iters = round.(Int, logrange(1, 10_000, length=50))
n_mc = 100
mc_trials = map(iters) do iter
    println("$iter \t / $(length(iters))")
    planner = AlphaZeroPlanner(game, oracle, max_iter=iter, c=10.0)
    Flux.loadmodel!(planner, joinpath(@modeldir, "oracle0100.jld2"))
    return map(1:n_mc) do i
        σ, info = behavior_info(planner, s)
        mat = AZ.node_matrix_game(info.tree, 1.0, 1, discount(game))
        x,y,v = solve(AZ.PATHSolver(), mat)
        return v
    end
end

trial_mat = reduce(hcat, mc_trials)' |> Array
means = vec(mean(trial_mat, dims=2))
stds = vec(std(trial_mat, dims=2))

scatter(iters, trial_mat, c=2, alpha=0.25, markerstrokewidth = 0)
plot!(iters, means, fillrange=(means .- stds, means .+ stds), c=1, alpha=0.50, lw=5)


##
using BenchmarkTools

include(joinpath(@__DIR__, "llbr.jl"))
π1 = ExperimentTools.policy1_from_oracle(planner.oracle)  # or your own policy function
s = rand(initialstate(game))
val = approx_br_value(game, planner.oracle, π1, s0; max_depth=7)
bm = @benchmark approx_br_value(game, planner.oracle, π1, s0; max_depth=7)
@code_warntype approx_br_value(game, planner.oracle, π1, s0; max_depth=7)

# If you also want the (limited-lookahead) BR actions P2 chose along visited states:
# val2, br_pol = approx_br_value(game, params.oracle, π1, s0; max_depth=6, return_policy=true)

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = policy1_from_oracle(planner.oracle)  # or your own policy function
    approx_br_value(game, planner.oracle, π1, s0; max_depth=10)
end

steps_per_iter = 10_000
iter = eachindex(readdir(@modeldir)) .- 1
steps = iter .* steps_per_iter

plot(steps, br_vals, xlabel="steps", ylabel="BRV", lw=2, title="Discrete Tag")

using ExperimentTools

@benchmark ExperimentTools.approx_br_value_mt(game, oracle, π1, s; parallel_depth=2)
ExperimentTools.approx_br_value(game, oracle, π1, s)

using JET

@code_warntype ExperimentTools.approx_br_value_mt(game, oracle, π1, s; parallel_depth=2)
ExperimentTools.approx_br_value_mt(game, oracle, π1, s; parallel_depth=3, max_depth=6)
@code_warntype ExperimentTools.approx_br_value_mt(game, oracle, π1, s; parallel_depth=2)
@code_warntype ExperimentTools.approx_br_value(game, oracle, π1, s; max_depth=3)
JET.@report_opt ExperimentTools.approx_br_value(game, oracle, π1, s; max_depth=3)
ExperimentTools.approx_br_value(game, oracle, π1, s; max_depth=3)


σ, info = behavior_info(planner, s)
tree.s


function policy1_from_tree(game, planner, tree)
    γ = discount(game)
    (;matrix_solver) = planner
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(AZ.node_matrix_game(tree, 1.0, idx, γ))
            x,y = AZ.state_policy(planner.oracle, game, s)
            return x
        end
        A = AZ.node_matrix_game(tree, 1.0, idx, γ)
        x,y,t = solve(matrix_solver, A)
        return x
    end
end


br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    σ, info = behavior_info(planner, s)
    π1 = policy1_from_tree(game, planner, info.tree)
    # π1 = policy1_from_oracle(planner.oracle)  # or your own policy function
    ExperimentTools.approx_br_value_mt(game, planner.oracle, π1, s; max_depth=10)
end


plot(br_vals)
