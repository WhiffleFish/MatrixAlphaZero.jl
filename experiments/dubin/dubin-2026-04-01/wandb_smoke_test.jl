using Pkg
Pkg.activate("experiments")

using MatrixAlphaZero
using ExperimentTools
using MarkovGames
using Flux
using POMDPs
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays

const AZ = MatrixAlphaZero

game = DubinMG(V = (1.0, 1.0))
na1, na2 = length.(actions(game))

trunk  = Chain(Dense(8, 16, tanh), Dense(16, 16, tanh))
actor  = AZ.MultiActor(Chain(Dense(16, na1)), Chain(Dense(16, na2)))
critic = Dense(16, 1)
oracle = AZ.ActorCritic(trunk, actor, critic)

s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

cb = (
    AZ.MetricsCallback(),
    WandbCallback(
        project = "Matrix AlphaZero",
        name    = "dubin-smoke",
        config  = Dict(
            "tree_queries" => 100,
            "max_depth"    => 3,
            "search_style" => "RegretMatchingSearch",
        ),
    ),
)

sol = AZ.AlphaZeroSolver(
    oracle        = oracle,
    max_steps     = 1_000,
    num_steps     = 1_000,
    update_epochs = 1,
    num_batches   = 8,
    mcts_params   = AZ.MCTSParams(;
        tree_queries = 100,
        max_depth    = 50,
        oracle,
    ),
)

get(ENV, "WANDB_API_KEY", "") != "" || error(
    "WANDB_API_KEY is not set. Export it before running:\n" *
    "  export WANDB_API_KEY=\"your_key_here\"\n" *
    "or find your key at https://wandb.ai/authorize"
)

@info "Running smoke test — check https://wandb.ai for live metrics"
solve(sol, game; s0 = Deterministic(s0), cb)
close(cb[2])
@info "Done. Verify that 3 steps appeared in the wandb run."
