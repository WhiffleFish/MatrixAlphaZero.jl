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
    using Pkg
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
end

EXPR_PATH = abspath(joinpath(dirname(@__DIR__), "..", "..", "experiments"))

Pkg.activate(EXPR_PATH)
println(Pkg.project().path)

using SDAGames.SNRGame
@everywhere begin
    Pkg.activate(abspath(joinpath(dirname(@__DIR__), "..", "..", "experiments")))
    println(Pkg.project().path)
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
trunk = Chain(Dense(15, 32, tanh), Dense(32, 32, tanh))
critic = AZ.HLGaussCritic(
    Chain(Dense(32, 32, tanh), Dense(32, 64)),
    -50, 50, 64
)
actor = MultiActor(
    Chain(Dense(32, 32, tanh), Dense(32, 16, tanh), Dense(16, 3)), 
    Chain(Dense(32, 32, tanh), Dense(32, 16, tanh), Dense(16, 3))
)

oracle = ActorCritic(trunk, actor, critic)
jldsave(joinpath(@__DIR__, "oracle.jld2"); oracle)

sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle=oracle, steps_per_iter=steps_per_iter, max_iter=iter, 
    lr = 3f-4,
    train_intensity = 6,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries = tree_queries, 
        oracle, 
        max_depth=max_depth, 
        temperature=t -> 1.0 * (0.90 ^ (t-1))
    )
)

cb = AZ.ModelSaveCallback(@modeldir)
pol, info = solve(sol, game; s0=initialstate(game), cb)
JLD2.jldsave(joinpath(@__DIR__, "train_info.jld2"); info...)

rmprocs(p)
