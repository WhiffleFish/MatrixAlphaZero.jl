using Test
using MatrixAlphaZero

const AZ = MatrixAlphaZero

include("support/fixtures.jl")
using .Fixtures

include("callbacks_tests.jl")
include("nn_tests.jl")
include("tree_mcts_tests.jl")
include("mcts_search_tests.jl")
include("solver_train_tests.jl")
include("mcts_solver_tests.jl")
include("critic_only_tests.jl")
include("vis_tests.jl")
