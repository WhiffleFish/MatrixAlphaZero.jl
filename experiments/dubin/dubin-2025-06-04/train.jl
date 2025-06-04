using Distributed
using JLD2
using ExperimentTools
args = ExperimentTools.parse_commandline()

p = addprocs(args["addprocs"])

@everywhere begin
    using MarkovGames
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Flux
    using POMDPTools
    using Distributions
    using POSGModels.Dubin
end

b0 = let
    s1 = Dubin.Vec3(0, 0, deg2rad(45))
    s2 = Dubin.Vec3(5, 5, deg2rad(45 + 180))
    Deterministic(JointDubinState(s1, s2))
end

game = DubinMG()
init = Flux.orthogonal(; gain=sqrt(2))
trunk = Chain(Dense(8, 32, tanh; init), Dense(32, 32, tanh; init))
critic = AZ.HLGaussCritic(
    Chain(Dense(32, 32, tanh; init), Dense(32, 32; init)),
    -15, 15, 32
)
actor = MultiActor(
    Chain(Dense(32, 16, tanh; init), Dense(16, 3; init)), 
    Chain(Dense(32, 16, tanh; init), Dense(16, 3; init))
)
oracle = ActorCritic(trunk, actor, critic)

jldsave(joinpath(@__DIR__, "oracle.jld2"); oracle)

sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle=oracle, steps_per_iter=10_000, max_iter=20, train_intensity=10,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries= 20, 
        oracle, 
        max_depth   = 50,
        temperature = t -> 1.0 * (0.90 ^ (t-1)),
        c           = 10.0
    )
)

cb = AZ.ModelSaveCallback(@modeldir)
pol, info = solve(sol, game; s0=b0, cb)
JLD2.jldsave(joinpath(@__DIR__, "train_info.jld2"); info...)
