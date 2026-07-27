using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributions
using POMDPTools
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics

include(joinpath(@__DIR__, "initial_state.jl"))
include(joinpath(@__DIR__, "..", "..", "approximate_shapley_residual.jl"))

# Full fixed-bank sweep:
# julia --project=experiments \
#   experiments/sda/sda-2026-07-21/approximate_shapley_residual.jl
# Uses up to four worker processes by default; pass --workers 0 for serial.
#
# Exact moving on-policy diagnostic:
# julia --project=experiments \
#   experiments/sda/sda-2026-07-21/approximate_shapley_residual.jl \
#   --state-mode on-policy

const SDA_RESIDUAL_SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const SDA_RESIDUAL_GAME = SNRGameSimple(altitude_bounds=(100e3, 2e7))
const SDA_RESIDUAL_CONFIG = (;
    name="sda-2026-07-21",
    runner_file=abspath(@__FILE__),
    game=SDA_RESIDUAL_GAME,
    initialstate_distribution=core_initialstate_distribution(SDA_RESIDUAL_GAME),
    initial_distribution_name=CORE_DISTRIBUTION_NAME,
    oracle_file=joinpath(
        @__DIR__,
        "oracle_$(SDA_RESIDUAL_SEARCH_NAME).jld2",
    ),
    models_dir=joinpath(
        @__DIR__,
        "models_$(SDA_RESIDUAL_SEARCH_NAME)",
    ),
    output_dir=joinpath(@__DIR__, "approximate_shapley_residual"),
    tree_queries=500,
    max_depth=5,
    sim_depth=50,
    backup=:mean,
    epsilon_schedule=iteration ->
        max(0.3 * (1 - 1e-3)^(iteration - 1), 0.1),
    state_dim=16,
)

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    run_approximate_shapley_residual(SDA_RESIDUAL_CONFIG)
