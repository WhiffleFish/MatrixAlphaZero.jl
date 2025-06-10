using Distributed
using ExperimentTools

args = ExperimentTools.parse_commandline(
    iter = 100,
    tree_queries = 100,
    max_depth = 50
)

p = addprocs(args["addprocs"])
tree_queries = args["tree_queries"]

@everywhere begin
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using JLD2
    using Flux
    using MarkovGames
    using POMDPs
    using POMDPTools
    using POSGModels.DiscreteTag
    using MCTS
    using ExperimentTools
end

game = TagMG(reward_model=DiscreteTag.DenseReward(peak=1.0))
oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle, max_iter=tree_queries, c=10.0)
mcts_solver = MCTSSolver(n_iterations=tree_queries)
res = ExperimentTools.exploitability(
    game, 
    @__DIR__; 
    n           = args["iter"], 
    max_steps   = args["max_depth"], 
    parallel    = true,
    planner, 
    mcts_solver
)
ExperimentTools.save_mats(res, joinpath(@__DIR__, "brv"))
rmprocs(p)
