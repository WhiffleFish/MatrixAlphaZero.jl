using Distributed
using JLD2
using ExperimentTools

args = ExperimentTools.parse_commandline(
    iter = 100,
    steps_per_iter = 10_000,
    tree_queries = 100,
    max_depth = 50
)

p = addprocs(args["addprocs"])
iter = args["iter"]
tree_queries = args["tree_queries"]
steps_per_iter = args["steps_per_iter"]
max_depth = args["max_depth"]

@everywhere begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using POSGModels.Dubin
end

b0 = ImplicitDistribution() do rng
    s1 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    s2 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    return JointDubinState(s1, s2)
end

game = DubinMG(V = (1.0, 1.0))
trunk = Chain(Dense(8, 16, tanh), Dense(16, 16, tanh))
critic = AZ.HLGaussCritic(
    Chain(Dense(16, 16, tanh), Dense(16, 32)),
    -10, 20, 32
)
na1, na2 = length.(actions(game))
actor = MultiActor(
    Chain(Dense(16, 32, tanh), Dense(32, na1)), 
    Chain(Dense(16, 32, tanh), Dense(32, na2))
)
oracle = ActorCritic(trunk, actor, critic)
jldsave(joinpath(@__DIR__, "oracle.jld2"); oracle)

sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle=oracle, steps_per_iter=steps_per_iter, max_iter=iter,
    buff_cap = 100_000,
    lr = 3f-4,
    train_intensity = 3,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries= tree_queries, 
        oracle, 
        max_depth   = max_depth,
        temperature = t -> 1.0 * (0.90 ^ (t-1)),
        c           = 10.0
    )
)

cb = AZ.ModelSaveCallback(@modeldir)
pol, info = solve(sol, game; s0=b0, cb)
JLD2.jldsave(joinpath(@__DIR__, "train_info.jld2"); info...)

rmprocs(p)
