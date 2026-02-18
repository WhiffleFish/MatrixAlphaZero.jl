using Distributed
using JLD2
using ExperimentTools

args = ExperimentTools.parse_commandline(
    iter = 50,
    steps_per_iter = 150_000,
    tree_queries = 150,
    max_depth = 50
)

p = addprocs(args["addprocs"])
iter = args["iter"]
tree_queries = args["tree_queries"]
steps_per_iter = args["steps_per_iter"]
max_depth = args["max_depth"]

# Stability-focused defaults
width = 32
lr = 3f-4
train_intensity = 2
ema_decay = 0.99f0
ema_selfplay = true
ema_callback = true

@everywhere begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using POSGModels.Dubin
    using POSGModels.StaticArrays
end

game = DubinMG(V = (1.0, 1.0))
trunk = Chain(Dense(8, width, tanh), Dense(width, width, tanh))
critic = AZ.HLGaussCritic(
    Chain(Dense(width, width, tanh), Dense(width, 32)),
    -10, 20, 32
)
na1, na2 = length.(actions(game))
actor = MultiActor(
    Chain(Dense(width, width, tanh), Dense(width, na1)), 
    Chain(Dense(width, width, tanh), Dense(width, na2))
)
oracle = ActorCritic(trunk, actor, critic)
jldsave(joinpath(@__DIR__, "oracle.jld2"); oracle)

sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle=oracle, steps_per_iter=steps_per_iter, max_iter=iter,
    buff_cap = 100_000,
    lr = lr,
    train_intensity = train_intensity,
    ema_decay = ema_decay,
    ema_selfplay = ema_selfplay,
    ema_callback = ema_callback,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries= tree_queries, 
        oracle, 
        max_depth   = max_depth,
        temperature = t -> 1.0 * (0.90 ^ (t-1)),
        c           = 10.0
    )
)

cb = AZ.ModelSaveCallback(@modeldir)
s0 = JointDubinState(SA[1,1,deg2rad(45)], SA[8,7,deg2rad(180)])
pol, info = solve(sol, game; s0=Deterministic(s0), cb)
JLD2.jldsave(joinpath(@__DIR__, "train_info.jld2"); info...)

rmprocs(p)
