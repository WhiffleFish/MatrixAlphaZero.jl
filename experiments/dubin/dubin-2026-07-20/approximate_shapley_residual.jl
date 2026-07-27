using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributions
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays

include(joinpath(@__DIR__, "..", "..", "approximate_shapley_residual.jl"))

# Full fixed-bank sweep:
# julia --project=experiments \
#   experiments/dubin/dubin-2026-07-20/approximate_shapley_residual.jl
# Uses up to four worker processes by default; pass --workers 0 for serial.
#
# Exact moving on-policy diagnostic:
# julia --project=experiments \
#   experiments/dubin/dubin-2026-07-20/approximate_shapley_residual.jl \
#   --state-mode on-policy

const DUBIN_RESIDUAL_SEARCH_NAME = "rm_plus_no_transfer_train"
const DUBIN_RESIDUAL_GAME = DubinMG(V=(1.0, 1.0))
const DUBIN_RESIDUAL_INITIAL_STATE = JointDubinState(
    SA[1, 1, deg2rad(45)],
    SA[8, 7, deg2rad(180)],
)
const DUBIN_RESIDUAL_CONFIG = (;
    name="dubin-2026-07-20",
    runner_file=abspath(@__FILE__),
    game=DUBIN_RESIDUAL_GAME,
    initialstate_distribution=Deterministic(DUBIN_RESIDUAL_INITIAL_STATE),
    initial_distribution_name="deterministic_reference_state",
    oracle_file=joinpath(
        @__DIR__,
        "oracle_$(DUBIN_RESIDUAL_SEARCH_NAME).jld2",
    ),
    models_dir=joinpath(
        @__DIR__,
        "models_$(DUBIN_RESIDUAL_SEARCH_NAME)",
    ),
    output_dir=joinpath(@__DIR__, "approximate_shapley_residual"),
    tree_queries=500,
    max_depth=5,
    sim_depth=50,
    backup=:sample,
    epsilon_schedule=iteration ->
        max(0.3 * (1 - 1e-3)^(iteration - 1), 0.1),
    state_dim=8,
)

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    run_approximate_shapley_residual(DUBIN_RESIDUAL_CONFIG)
