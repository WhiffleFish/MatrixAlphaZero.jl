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
SV = mapreduce(hcat, S) do s
    MarkovGames.convert_s(Vector{Float32}, s, game)
end

p1, p2 = AZ.policy(oracle, SV)
using Statistics
p1_ents = map(eachcol(p1)) do col
    -sum(col .* log.(col))
end
histogram(p1_ents)

p1
