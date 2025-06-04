begin
    using Pkg
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
mcts_solver = MCTSSolver(n_iterations=10)
res = ExperimentTools.exploitability(game, @__DIR__; n = 5, max_steps=10, planner, mcts_solver, parallel=false)
res[1]

using DelimitedFiles

ExperimentTools.save_mats(res, joinpath(@__DIR__, "brv"))


modeldir = @modeldir
basedir = @__DIR__
mcts_solver = MCTSSolver(n_iterations=10)
modeldir = joinpath(basedir, "models")
model_path = readdir(modeldir, join=true)[1]
oracle = AZ.load_oracle(basedir)
planner = AlphaZeroPlanner(game, oracle)
Flux.loadmodel!(planner, model_path)

max_steps = 30
n = 100
s0 = initialstate(game)
mcts_az_params      = (;max_iter=0)
sims1 = ExperimentTools.ExploitabilitySim(
    [RolloutSimulator(;max_steps, rng=Random.default_rng()) for _ in 1:n],
    game,
    mcts_solver,
    planner;
    mcts_az_params,
    s0 = ExperimentTools.dist2states(Random.default_rng(), game, s0, n),
    mcts_player = 1
)
res1 = run(sims1)

sim = sims1[1]
@which simulate(sim.simulator, sim.mg, sim.policy, sim.initialstate)

sim.policy.pols[1]

POMDPs.simulate(sims1[1])




ExperimentTools.evaluate(game, mcts_solver, planner)

modeldir = joinpath(basedir, "models")

model_path = readdir(modeldir, join=true)[1]
oracle = AZ.load_oracle(basedir)
planner = AlphaZeroPlanner(oracle, game)
Flux.loadmodel!(planner, model_path)


ExperimentTools.exploitability(game, @__DIR__)

i = 19
model_state = JLD2.load(
    joinpath(@__DIR__, "models", "oracle"*AZ.iter2string(i)*".jld2"), 
    "model_state"
)
Flux.loadmodel!(oracle, model_state)

begin
    s_defender = Dubin.Vec3(7, 1, π)
    s_attackers = map(1:10_000) do i
        Dubin.Vec3(rand()*game.floor[1], rand()*game.floor[2], deg2rad(0))
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
