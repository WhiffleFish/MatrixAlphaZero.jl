using Distributed
using ExperimentTools
args = ExperimentTools.parse_commandline()

p = addprocs(args["addprocs"])

@everywhere begin
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using JLD2
    using Flux
    using MarkovGames
    using POMDPs
    using POMDPTools
    using POSGModels.Dubin
    using MCTS
    using ExperimentTools
end

b0 = ImplicitDistribution() do rng
    s1 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    s2 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    return JointDubinState(s1, s2)
end

game = DubinMG()
oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
mcts_solver = MCTSSolver(n_iterations=100)
res = ExperimentTools.exploitability(
    game, 
    @__DIR__; 
    n           = 100, 
    max_steps   = 50, 
    parallel    = true,
    planner, 
    mcts_solver
)
ExperimentTools.save_mats(res, joinpath(@__DIR__, "brv-c10"))
rmprocs(p)
