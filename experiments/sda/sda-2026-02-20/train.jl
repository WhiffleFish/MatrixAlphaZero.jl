using Distributed
using JLD2
using ExperimentTools

args = ExperimentTools.parse_commandline(
    iter = 100,
    steps_per_iter = 100_000,
    tree_queries = 150,
    max_depth = 50
)

p = addprocs(args["addprocs"])
iter = args["iter"]
tree_queries = args["tree_queries"]
steps_per_iter = args["steps_per_iter"]
max_depth = args["max_depth"]

width = 32
lr = 3f-4
train_intensity = 1
ema_decay = 0.99f0

@everywhere begin
    using Pkg
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using SDAGames.SNRGame
    using SDAGames.SatelliteDynamics
end

d_observer = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

d_target = ImplicitDistribution() do rng
    sOSCtoCART([
        R_EARTH .+ rand(rng, Distributions.Uniform(500e3, 1e7)),
        0.0,
        0.0,
        0.0,
        0.0,
        rand(rng) * 2π
    ])
end

game = SNRSDAGame(observer=d_observer, target=d_target, altitude_bounds=(100e3, 2e7))
trunk = Chain(Dense(15, width, tanh), Dense(width, width, tanh))
critic = AZ.HLGaussCritic(
    Chain(Dense(width, width, tanh), Dense(width, width)),
    -50, 50, width
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
    buff_cap = 1_000_000,
    batchsize = 256,
    lr = lr,
    train_intensity = train_intensity,
    ema_decay = ema_decay,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries= tree_queries, 
        oracle, 
        max_depth   = max_depth,
        matrix_solver = MatrixAlphaZero.RegretSolver(100),
        c           = 10.0
    )
)

cb = AZ.ModelSaveCallback(@modeldir)
pol, info = solve(sol, game; s0=initialstate(game), cb)
JLD2.jldsave(joinpath(@__DIR__, "train_info.jld2"); info...)

rmprocs(p)
