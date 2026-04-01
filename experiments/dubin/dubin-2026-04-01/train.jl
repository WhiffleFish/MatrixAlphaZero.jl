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
    (; name = "matrix_game", label = "Greedy Matrix", style = AZ.MatrixGameSearch(c=10.0, matrix_solver=AZ.RegretSolver(100))),
    (; name = "regret_matching", label = "Regret Matching", style = AZ.RegretMatchingSearch()),
    (; name = "exp3", label = "Exp3", style = AZ.Exp3Search()),
)

args = ExperimentTools.parse_commandline(
    iter = 50,
    steps_per_iter = 100_000,
    tree_queries = 150,
    max_depth = 50,
)

p = addprocs(args["addprocs"])
iter = args["iter"]
tree_queries = args["tree_queries"]
steps_per_iter = args["steps_per_iter"]
max_depth = args["max_depth"]

# Stability-focused defaults
width = 32
lr = 3f-4
train_intensity = 1
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

for spec in STYLE_SPECS
    style_dir = joinpath(@__DIR__, spec.name)
    models_dir = joinpath(style_dir, "models")
    mkpath(style_dir)

    oracle = deepcopy(base_oracle)
    jldsave(joinpath(style_dir, "oracle.jld2"); oracle)

    sol = MatrixAlphaZero.AlphaZeroSolver(
        oracle = oracle,
        steps_per_iter = steps_per_iter,
        max_iter = iter,
        buff_cap = 1_000_000,
        batchsize = 256,
        lr = lr,
        train_intensity = train_intensity,
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
            config  = Dict(
                "search_style"   => spec.name,
                "tree_queries"   => tree_queries,
                "max_depth"      => max_depth,
                "steps_per_iter" => steps_per_iter,
                "iter"           => iter,
                "width"          => width,
                "lr"             => lr,
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
