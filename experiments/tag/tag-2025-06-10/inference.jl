begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Plots
    using POSGModels.DiscreteTag
    using Flux
    using POMDPTools
    using POMDPs
end

game = TagMG(reward_model=DiscreteTag.DenseReward(peak=1.0))
oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
Flux.loadmodel!(planner, @modeldir("oracle0040.jld2"))

sim = HistoryRecorder(max_steps=50)
hist = simulate(sim, game, planner)

anim = @animate for h_i in hist
    plot(game, h_i[:s], h_i[:behavior])
end

gif(anim, @figdir("sim-1000iter.gif"), fps=2)



##
using JLD2
using Plots
info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
ks = keys(info)

plot(reduce(vcat,info["train_losses"]))
plot(reduce(vcat,info["value_losses"]))
plot(reduce(vcat,info["policy_losses"]))

## simulate

s = TagState(Coord(2,2), Coord(1,1))
to_f32(s) = convert_s(Vector{Float32}, s, game)
s_v = to_f32(s)

map(0:40) do i

    Flux.loadmodel!(planner, @modeldir("oracle0040.jld2")) 
end

s_idx = stateindex(game, s)
p1_policies, p2_policies = getindex.(P_az_full, 1), getindex.(P_az_full, 2)

map(p1_policies) do p1_pol
    p1_pol[:,s_idx]
end

s_pol2 = map(p2_policies) do p2_pol
    p2_pol[:,s_idx]
end

anim = @animate for i ∈ 1:41
    bar(["up", "right", "down", "left"], s_pol2[i], title=i-1)
end

# lots of seeminly random policy changes - regularize somehow to stop this random walk?
gif(anim, @figdir("policy-progression.gif"), fps=3)

anim = @animate for i ∈ eachindex(V_az_full)
    histogram(V_az_full[i], title="$(i)", bins=0:20)
end

gif(anim, @figdir("value-progression.gif"), fps=5)

p1_diffs = map(2:41) do i
    Flux.crossentropy(p1_policies[i], p1_policies[i-1])
end

p2_diffs = map(2:41) do i
    Flux.crossentropy(p2_policies[i], p2_policies[i-1])
end

using LaTeXStrings
plot(
    1:40, 
    [p1_diffs p2_diffs], 
    lw=2, 
    label=["Player 1" "Player 2"],
    title = "Policy Crossentropy Change",
    xlabel = "AlphaZero iteration (i)",
    ylabel = L"CE(\pi_i, \pi_{i-1})"
)

savefig(@figdir("policy-crossentropy-change.png"))
savefig(@figdir("policy-crossentropy-change.pdf"))
