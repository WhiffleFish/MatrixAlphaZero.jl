using Pkg
Pkg.activate("experiments")

using Distributed
using JLD2
using ExperimentTools
using MatrixAlphaZero
using Random

const AZ = MatrixAlphaZero
const SEARCH_NAME = "regret_matching"

args = ExperimentTools.parse_commandline(
    max_steps = 5_000_000,
    num_steps = 100_000,
    update_epochs = 1,
    num_batches = 400,
    tree_queries = 150,
    max_depth = 50,
)

p = addprocs(args["addprocs"])
max_steps = args["max_steps"]
num_steps = args["num_steps"]
update_epochs = args["update_epochs"]
num_batches = args["num_batches"]
tree_queries = args["tree_queries"]
max_depth = args["max_depth"]

# Stability-focused defaults
width = 32
lr = 3f-4
ema_decay = 0.99f0

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
na1, na2 = length.(actions(game))

function init_oracle(width, na1, na2)
    trunk = Chain(Dense(8, width, tanh), Dense(width, width, tanh))
    actor = MultiActor(
        Chain(Dense(width, width, tanh), Dense(width, na1)),
        Chain(Dense(width, width, tanh), Dense(width, na2)),
    )
    critic = AZ.HLGaussCritic(
        Chain(Dense(width, width, tanh), Dense(width, 32)),
        -10, 20, 32,
    )
    return ActorCritic(trunk, actor, critic)
end

Random.seed!(0)
base_oracle = init_oracle(width, na1, na2)
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

style_dir = joinpath(@__DIR__, SEARCH_NAME)
models_dir = joinpath(style_dir, "models")
mkpath(style_dir)

oracle = deepcopy(base_oracle)
jldsave(joinpath(style_dir, "oracle.jld2"); oracle)

sol = MatrixAlphaZero.AlphaZeroSolver(
    oracle = oracle,
    max_steps = max_steps,
    num_steps = num_steps,
    update_epochs = update_epochs,
    num_batches = num_batches,
    lr = lr,
    ema_decay = ema_decay,
    mcts_params = MatrixAlphaZero.MCTSParams(;
        tree_queries = tree_queries,
        oracle,
        max_depth = max_depth,
    ),
)

cb = AZ.ModelSaveCallback(models_dir)
solve(sol, game; s0 = Deterministic(s0), cb)

rmprocs(p)
