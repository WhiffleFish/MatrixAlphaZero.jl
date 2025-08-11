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



game = DubinMG()
oracle = AZ.load_oracle(@__DIR__)
iter = 1000
planner = AlphaZeroPlanner(game, oracle, max_iter=iter, c=10.0)
Flux.loadmodel!(planner, @modeldir("oracle0100.jld2"))

using POSGModels.StaticArrays
s = JointDubinState(SA[1,1,deg2rad(45)], SA[2,2,deg2rad(45 + 180)])

sim = HistoryRecorder(max_steps=50)
hist = simulate(sim, game, planner)

anim = @animate for h_i in hist
    plot(game, h_i[:s], h_i[:behavior])
end

gif(anim, @figdir("sim-$(iter)iter2.gif"), fps=2)

plot(collect(hist[:r]))

Dubin.closest_distance(hist[1].s, hist[2].s)

s = rand(initialstate(game))
@edit isterminal(game,  s)

b, info = behavior_info(planner, s)
γ = discount(game)
solve(info.tree.r[1] + γ * info.tree.v[1])


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

## simulate

s = TagState(Coord(2,2), Coord(1,1))
to_f32(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
s_v = to_f32(s)

model_nums = eachindex(readdir(@modeldir)) .- 1

s_idx = POMDPs.stateindex(game, s)

P_az_full = map(states(game)) do s
    AZ.state_policy(oracle, game, s)
end

P_az_full = map(0:50) do i
    n = AZ.iter2string(i)
    model_str = "oracle" * n * ".jld2"
    Flux.loadmodel!(planner, @modeldir(model_str))
    state_pols = map(states(game)) do s
        AZ.state_policy(oracle, game, s)
    end
    reduce(hcat, getindex.(state_pols, 1)), reduce(hcat, getindex.(state_pols, 2))
end

p1_policies, p2_policies = getindex.(P_az_full, 1), getindex.(P_az_full, 2)


model_str = "oracle0000.jld2"
@modeldir(model_str)

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


model_nums = 0:50
p1_diffs = map(2:lastindex(model_nums)) do i
    Flux.crossentropy(p1_policies[i], p1_policies[i-1])
end

p2_diffs = map(2:lastindex(model_nums)) do i
    Flux.crossentropy(p2_policies[i], p2_policies[i-1])
end

using LaTeXStrings
plot(
    1:(lastindex(model_nums)-1),
    [p1_diffs p2_diffs], 
    lw=2, 
    label=["Player 1" "Player 2"],
    title = "Policy Crossentropy Change",
    xlabel = "AlphaZero iteration (i)",
    ylabel = L"CE(\pi_i, \pi_{i-1})"
)

savefig(@figdir("policy-crossentropy-change.png"))
savefig(@figdir("policy-crossentropy-change.pdf"))

info = jldopen(joinpath(@__DIR__,"train_info.jld2"))
info["buffer"]

##
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
using DelimitedFiles
using ExperimentTools

br1 = readdlm(joinpath(@__DIR__, "brv", "br1.csv"), ',')
br2 = readdlm(joinpath(@__DIR__, "brv", "br2.csv"), ',')

plot(br1)
plot(br2)

ed = ExploitabilityData(joinpath(@__DIR__, "brv"))
plot(ed)
