# ExperimentTools

[incomplete] Usage
```julia
using MatrixAlphaZero
using ExperimentTools
const AZ = MatrixAlphaZero

oracle = AZ.load_oracle(@__DIR__)
planner = AlphaZeroPlanner(game, oracle)
mcts_solver = MCTSSolver(n_iterators = 100)
res = ExperimentTools.exploitability(
    game, @__DIR__; 
    n           = 2, 
    max_steps   = 10,
    parallel    = true
    mcts_solver,
    planner,
)
```
