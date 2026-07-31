using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributed
using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random

const AZ = MatrixAlphaZero
const SEED = 0
const SEARCH_NAME = "rm_plus_no_transfer_train"

args = ExperimentTools.parse_commandline(
    max_steps=10_000_000,
    num_steps=8192,
    update_epochs=2,
    num_batches=4,
    tree_queries=500,
    max_depth=5,
    sim_depth=50,
)
workers = addprocs(args["addprocs"])

@everywhere begin
    using Flux
    using MarkovGames
    using MatrixAlphaZero
    using POMDPTools
    using POSGModels.Dubin
    using POSGModels.StaticArrays
    using Random
end
@everywhere Random.seed!($SEED + Distributed.myid())

function state_network(width, output_dim)
    return Chain(
        Dense(8 => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
        Dense(width => output_dim),
    )
end

function make_oracle(width, na1, na2)
    return AZ.FittedRegretModel(
        AZ.MultiActor(state_network(width, na1), state_network(width, na2)),
        AZ.MultiActor(state_network(width, na1), state_network(width, na2)),
        state_network(width, 1);
        value_weight=1.0f0,
        regret_weight=0.1f0,
        strategy_weight=0.5f0,
    )
end

game = DubinMG(V=(1.0, 1.0))
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
na1, na2 = length.(actions(game))

Random.seed!(SEED)
oracle = make_oracle(32, na1, na2)
Random.seed!(SEED + 1_000_000)
epsilon_decay = 1 - 1e-3
epsilon_schedule = update -> max(0.3 * epsilon_decay^(update - 1), 0.1)
search = AZ.MCTSSearch(;
    oracle,
    tree_queries=args["tree_queries"],
    max_depth=args["max_depth"],
    search_style=AZ.RegretMatchingSearch(; backup=:sample, method=AZ.Plus()),
    value_target=:search,
    ϵ=epsilon_schedule,
    prior_scale=0.0,
)
solver = AZ.AlphaZeroSolver(;
    search,
    max_steps=args["max_steps"],
    num_steps=args["num_steps"],
    sim_depth=args["sim_depth"],
    update_epochs=args["update_epochs"],
    num_batches=args["num_batches"],
    lr=3f-4,
    lr_decay=0.999f0,
    lr_min=1f-5,
    lr_max=3f-4,
    ema=false,
    ema_decay=0.98f0,
    gae_lambda=0.95,
    rng=MersenneTwister(SEED),
)

output_dir = args["test"] ? mktempdir() : @__DIR__
models_dir = joinpath(output_dir, "models_$(SEARCH_NAME)")
mkpath(models_dir)
jldsave(joinpath(output_dir, "oracle_$(SEARCH_NAME).jld2"); oracle)

callbacks = (AZ.ModelSaveCallback(models_dir), AZ.MetricsCallback())
try
    solve(solver, game; s0=Deterministic(s0), cb=callbacks)
finally
    isempty(workers) || rmprocs(workers)
end
