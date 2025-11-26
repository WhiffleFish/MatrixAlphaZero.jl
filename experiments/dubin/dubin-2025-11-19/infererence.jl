begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Plots
    using POSGModels.Dubin
    using Flux
    using POMDPTools
    using POMDPs
end

game = DubinMG(V=(1.0, 1.0))
oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
iter = 100
planner = AlphaZeroPlanner(game, oracle, max_iter=iter, c=10.0)

using POSGModels.StaticArrays
s = JointDubinState(SA[1,1,deg2rad(45)], SA[2,2,deg2rad(45 + 180)])

sim = HistoryRecorder(max_steps=50)
hist = simulate(sim, game, planner)

anim = @animate for h_i in hist
    plot(game, h_i[:s], h_i[:behavior])
end

gif(anim, @figdir("sim-$(iter)iter.gif"), fps=2)

##
using JLD2
using Plots
using MarkovGames
using POMDPs
info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
ks = keys(info)

plot(reduce(vcat,info["train_losses"]))
plot(reduce(vcat,info["value_losses"]))
plot(reduce(vcat,info["policy_losses"]))


X = 0:0.1:10
Y = 0:0.1:10

for i ∈ 0:100
    oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle" * AZ.iter2string(i) * ".jld2"))
    V = map(Iterators.product(X,Y)) do (x,y)
        s = JointDubinState(SA[x,y, deg2rad(0)], SA[5,5,deg2rad(180)])
        AZ.value(oracle, POMDPs.convert_s(Vector{Float32}, s, game))
    end
    display(heatmap(X,Y,V))
end

oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle" * AZ.iter2string(i) * ".jld2"))

begin
    AZ.@model(1)
end

oracle = AZ.@model(1)
2
info["buffer"].s[1]

initialstate(game)

