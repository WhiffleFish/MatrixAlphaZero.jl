begin
    using Pkg
    Pkg.activate("experiments")
    using ExperimentTools
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Plots
    using JLD2
    using Flux
    using MarkovGames
    using POMDPs
    using POMDPTools
    using POSGModels.Dubin
    using MCTS
    using Plots
    using Random
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

b0 = ImplicitDistribution() do rng
    s1 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    s2 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    return JointDubinState(s1, s2)
end

game = DubinMG()
oracle = AZ.load_oracle(@__DIR__)

planner = AlphaZeroPlanner(game, oracle, max_iter=100)
Flux.loadmodel!(planner, last(readdir(@modeldir; join=true)))
mcts_solver = MCTSSolver(n_iterations=10)

function plot_dubin_value(oracle; 
        s_defender  = Dubin.Vec3(5, 5, π), 
        n           = 10_000,
        θ           = 0.0
    )
    s_attackers = map(1:n) do i
        Dubin.Vec3(rand()*game.floor[1], rand()*game.floor[2], deg2rad(θ))
    end
    attacker_coords = map(s_attackers) do s_a
        s_a[1], s_a[2]
    end
    full_states = map(s_attackers) do s_a
        Dubin.JointDubinState(s_a, s_defender)
    end
    vec_states = mapreduce(hcat, full_states) do fs
        MarkovGames.convert_s(Vector{Float32}, fs, game)
    end
    V = vec(oracle(vec_states).value)
    scatter(attacker_coords, ms=10, alpha=0.5, zcolor=Float64.(V), markerstrokewidth=0, label="")
    scatter!([s_defender[1]], [s_defender[2]], c=:green, ms=10, label="defender")
    return plot!(
        [s_defender[1], s_defender[1] + 1*cos(s_defender[3])],
        [s_defender[2], s_defender[2] + 1*sin(s_defender[3])],
        arrow=true
    )
end
anim = @animate for modelpath ∈ readdir(@modeldir; join=true)
    Flux.loadmodel!(planner, modelpath)
    plot_dubin_value(oracle, s_defender=Dubin.Vec3(3,5,π))
end

gif(anim, "dubin_values.gif", fps = 5)

Flux.loadmodel!(planner, last(readdir(@modeldir; join=true)))
anim = @animate for x ∈ reverse(1:0.1:10)
    plot_dubin_value(oracle, s_defender=Dubin.Vec3(x,5,π))
end
gif(anim, "dubin_values.gif", fps = 5)

begin
    s_defender = Dubin.Vec3(5, 5, π)
    s_attackers = map(1:10_000) do i
        Dubin.Vec3(rand()*game.floor[1], rand()*game.floor[2], deg2rad(θ))
    end
    attacker_coords = map(s_attackers) do s_a
        s_a[1], s_a[2]
    end
    full_states = map(s_attackers) do s_a
        Dubin.JointDubinState(s_a, s_defender)
    end
    vec_states = mapreduce(hcat, full_states) do fs
        MarkovGames.convert_s(Vector{Float32}, fs, game)
    end
    V = vec(oracle(vec_states).value)
    scatter(attacker_coords, ms=10, alpha=0.5, zcolor=Float64.(V), markerstrokewidth=0, label="")
    scatter!([s_defender[1]], [s_defender[2]], c=:green, ms=10, label="defender")
    plot!(
        [s_defender[1], s_defender[1] + 1*cos(s_defender[3])],
        [s_defender[2], s_defender[2] + 1*sin(s_defender[3])],
        arrow=true
    )
end

savefig(joinpath(@__DIR__, "figures", "attacker_value_map-heading0.pdf"))
savefig(joinpath(@__DIR__, "figures", "attacker_value_map-heading0.png"))



##

pol = AlphaZeroPlanner(oracle, game)
sim = RolloutSimulator(max_steps=50)
s = rand(initialstate(game))
simulate(sim, game, pol, s)



#
simulate(
    RolloutSimulator(max_steps=10),
    game,
    planner,
    rand(initialstate(game))
)
