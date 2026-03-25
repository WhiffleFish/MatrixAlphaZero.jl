using Test
using MatrixAlphaZero

const AZ = MatrixAlphaZero

include("support/fixtures.jl")
using .Fixtures

include("matrix_tests.jl")
include("buffer_tests.jl")
include("callbacks_tests.jl")
include("nn_tests.jl")
include("tree_mcts_tests.jl")
include("solver_train_tests.jl")
include("vis_tests.jl")
