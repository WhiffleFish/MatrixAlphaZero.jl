using Pkg
Pkg.activate("experiments")

using Distributed
using JLD2
using ExperimentTools
using MatrixAlphaZero
using MarkovGames
using Random

const AZ = MatrixAlphaZero
const STYLE_SPECS = (
    (; name = "regret_matching", label = "Regret Matching", style = AZ.RegretMatchingSearch()),
)

args = ExperimentTools.parse_commandline(
    max_steps = 5_000_000,
    num_steps = 100_000,
    update_epochs = 2,
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
lr_decay = 0.98f0      # LR halves roughly every 34 iterations
ema_decay = 0.98f0     # faster-tracking EMA reduces target lag

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

for spec in STYLE_SPECS
    style_dir = joinpath(@__DIR__, spec.name)
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
        lr_decay = lr_decay,
        ema_decay = ema_decay,
        mcts_params = MatrixAlphaZero.MCTSParams(;
            tree_queries = tree_queries,
            oracle,
            max_depth = max_depth,
            search_style = spec.style,
        ),
    )

    wandb_cb = if get(ENV, "WANDB_API_KEY", "") != ""
        WandbCallback(
            project = "Matrix AlphaZero",
            name    = "dubin-$(spec.name)",
            group   = "dubin-2026-04-02",
            config  = Dict(
                "search_style"   => spec.name,
                "tree_queries"   => tree_queries,
                "max_depth"      => max_depth,
                "max_steps"      => max_steps,
                "num_steps"      => num_steps,
                "update_epochs"  => update_epochs,
                "num_batches"    => num_batches,
                "width"          => width,
                "lr"             => lr,
                "lr_decay"       => lr_decay,
                "ema_decay"      => ema_decay,
            ),
        )
    else
        @warn "WANDB_API_KEY not set — skipping W&B logging for $(spec.name)"
        nothing
    end

    cb = if isnothing(wandb_cb)
        (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback())
    else
        (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback(), wandb_cb)
    end

    _, info = solve(sol, game; s0 = Deterministic(s0), cb)
    isnothing(wandb_cb) || close(wandb_cb)
    JLD2.jldsave(joinpath(style_dir, "train_info.jld2"); info...)
end

rmprocs(p)
