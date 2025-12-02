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
oracle = @model(50)
iter = 100
planner = AlphaZeroPlanner(game, oracle, max_iter=iter, c=10.0)

using POSGModels.StaticArrays
s = JointDubinState(SA[1,1,deg2rad(45)], SA[7,7,deg2rad(45 + 180)])
s = JointDubinState(SA[1,1,deg2rad(45)], SA[8,7,deg2rad(180)])

sim = HistoryRecorder(max_steps=50)
hist = simulate(sim, game, planner, s)

anim = @animate for h_i in hist
    plot(game, h_i[:s], h_i[:behavior])
end

gif(anim, @figdir("sim-$(iter)iter.gif"), fps=2)


X = 0:0.1:10
Y = 0:0.1:10

for i ∈ 0:50
    oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle" * AZ.iter2string(i) * ".jld2"))
    V = map(Iterators.product(X,Y)) do (x,y)
        s = JointDubinState(SA[x,y, deg2rad(0)], SA[5,5,deg2rad(180)])
        AZ.value(oracle, POMDPs.convert_s(Vector{Float32}, s, game))
    end
    display(heatmap(X,Y,V))
end

for _x ∈ 1:10
    V = map(Iterators.product(X,Y)) do (x,y)
        s = JointDubinState(SA[x,y, deg2rad(0)], SA[5,_x,deg2rad(180)])
        AZ.value(oracle, POMDPs.convert_s(Vector{Float32}, s, game))
    end
    display(heatmap(X,Y,V'))
    sleep(0.5)
end

θ = 0.0

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
savefig(@figdir("dubin_attacker_value_map.pdf"))

histogram(buf.v)

heatmap(X, Y, X * Y')

buf = info["buffer"]


oracle = @model(100)
sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle=oracle, steps_per_iter=10_000, max_iter=100,
    buff_cap = 100_000,
    lr = 1f-2,
    train_intensity = 3,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries= 20, 
        oracle, 
        max_depth   = 30,
        temperature = t -> 1.0 * (0.90 ^ (t-1)),
        c           = 10.0
    )
)

_info = AZ.train!(sol, oracle, buf, train_intensity=20, lr=0)
plot(_info.value_losses)

begin
    V = map(Iterators.product(X,Y)) do (x,y)
        s = JointDubinState(SA[x,y, deg2rad(0)], SA[5,5,deg2rad(180)])
        AZ.value(oracle, POMDPs.convert_s(Vector{Float32}, s, game))
    end
    display(heatmap(X,Y,V))
end
