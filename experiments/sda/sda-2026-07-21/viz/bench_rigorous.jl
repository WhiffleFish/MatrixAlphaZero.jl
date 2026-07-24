using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
using MarkovGames, MatrixAlphaZero, Flux, JLD2, POMDPTools, Distributions
using SDAGames.SNRGame, SDAGames.SatelliteDynamics, ExperimentTools, Random, Statistics
const AZ = MatrixAlphaZero; const Tools = ExperimentTools
const expdir = joinpath(@__DIR__, "..")
include(joinpath(expdir, "initial_state.jl"))

game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
oracle = AZ.load_oracle(joinpath(expdir, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
Flux.loadmodel!(oracle, JLD2.load(joinpath(expdir, "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"])

search = AZ.MCTSSearch(; oracle, tree_queries=100, max_depth=5, max_time=Inf,
    search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
    value_target=:search, ϵ=_->0.1, prior_scale=0.0)
planner = AZ.AlphaZeroPlanner(game, search)
observer = Tools.JointPolicy(Tools.SinglePlayerAlphaZeroPolicy(planner, 1), Tools.sda_no_burn_heuristic(game, 2))

s0 = core_initialstate_distribution(game)
Random.seed!(0)
init = [rand(s0) for _ in 1:8]

# one-sided rollout return: observer runs search, target coasts (passive)
function bench(n)
    t = @elapsed r = Tools.evaluate_joint_policy(game, observer, n;
        max_steps=50, initialstates=init[1:min(n,length(init))],
        show_progress=false, proc_warn=false, parallel=false)
    return t, r.reward[1]
end
bench(1)                      # warmup / compile
t8, _ = bench(8)
per_episode = t8 / 8
println("per passive-target rollout (depth 50, 100 queries): ", round(per_episode*1000; digits=1), " ms")
for (grid, frames, eps) in [(40*40,24,3),(60*60,24,3),(40*40,12,1)]
    total = grid*frames*eps*per_episode
    println("  grid=$(grid) frames=$(frames) eps=$(eps)  ->  ",
            round(total/60; digits=1), " min  (", round(total/3600; digits=2), " h)")
end
