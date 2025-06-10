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
    iter        = 20, 
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
planner = AlphaZeroPlanner(game, oracle)
V_az_full = map(0:40) do i
    Flux.loadmodel!(planner, joinpath(@modeldir, "oracle"*AZ.iter2string(i)*".jld2"))
    AZ.value(oracle, VS)
end
P_az_full = map(0:40) do i
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
savefig(joinpath(@__DIR__, "figures", "AZ-FP-policy-error.pdf"))
savefig(joinpath(@__DIR__, "figures", "AZ-FP-policy-error.png"))

plot(
    eachindex(value_mse) .- 1,
    value_mse,
    xlabel = "Training Iteration",
    ylabel = "Value MSE",
    title = "Discrete Tag AlphaZero / FP Comparison",
    lw = 2
)
savefig(joinpath(@__DIR__, "figures", "AZ-FP-value-mse.pdf"))
savefig(joinpath(@__DIR__, "figures", "AZ-FP-value-mse.png"))


## anim

pol_anim = @animate for pol_diff in pol_diffs
    plot(pol_diff, label="")
end

gif(pol_anim, joinpath(@__DIR__, "pol_anim.gif"), fps=3)

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
Flux.loadmodel!(planner, joinpath(@modeldir, "oracle0040.jld2"))

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
