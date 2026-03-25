using Random

@testset "tree.jl and mcts.jl" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        policies=Dict(0f0 => (Float32[0.8, 0.2], Float32[0.1, 0.9])),
    )
    tree = AZ.Tree(game, false)
    @test AZ.is_leaf(tree, 1)

    AZ.expand_s!(tree, 1, game, oracle)
    @test !AZ.is_leaf(tree, 1)
    @test size(tree.s_children[1]) == (2, 2)
    @test length(tree.s) == 5
    @test tree.r[1] == game.rewards
    @test tree.v[1] == zeros(2, 2)
    @test tree.prior[1][1] == Float32[0.8, 0.2]
    @test tree.prior[2][1] == Float32[0.1, 0.9]

    @test AZ.node_matrix_game(tree, 1, 1.0) == game.rewards
    @test AZ.node_matrix_game(tree, 0.0, 1, 1.0) == game.rewards

    tree.n_s[1] = 4
    tree.n_sa[1] .= [
        1 0
        0 3
    ]
    ucb = AZ.ucb_exploration(tree, 1.5, 1)
    @test size(ucb) == (2, 2)
    @test ucb[1, 2] > ucb[2, 2]

    pucb = AZ.pucb_exploration(tree, 1.5, 1; ϵ=0.0)
    @test size(pucb) == (2, 2)
    @test pucb[1, 2] > pucb[2, 1]

    @test AZ.ucb_matrix_games(tree, 1.5, 1, 1.0; ϵ=0.0)[1] == game.rewards .+ pucb
    @test AZ.uniform(3) == fill(1 / 3, 3)
    @test AZ.eps_exploration([1.0, 0.0], 0.2) ≈ [0.9, 0.1]
    @test AZ.zs_reward_scalar((3.0, -3.0)) == 3.0
    @test AZ.zs_reward_scalar(2.5) == 2.5

    Random.seed!(7)
    idx = AZ.action_idx_from_probs([0.0, 1.0], [1.0, 0.0])
    @test idx == CartesianIndex(2, 1)

    params = AZ.MCTSParams(
        tree_queries=1,
        c=0.5,
        max_depth=2,
        max_time=1.0,
        matrix_solver=Fixtures.GreedyMatrixSolver(),
        oracle=oracle,
    )
    (x, y, v), info = AZ.search_info(params, game, false; ϵ=0.0)
    @test x == [0.0, 1.0]
    @test y == [0.0, 1.0]
    @test v == 2.0
    @test length(info.tree.s) == 5
    @test AZ.search(params, game, false; ϵ=0.0) == (x, y, v)

    tree2 = AZ.Tree(game, false)
    @test AZ.simulate(params, tree2, game, 1; ϵ=0.0) == 2.0

    tree3 = AZ.Tree(game, false)
    AZ.expand_s!(tree3, 1, game, oracle)
    @test AZ.explore_action(Fixtures.GreedyMatrixSolver(), tree3, 0.5, 1, 1.0; ϵ=0.0) == CartesianIndex(2, 2)

    hist = AZ.mcts_sim(params, game, false; progress=false, ϵ=0.0)
    @test length(hist.s) == 1
    @test hist.r == [2.0]
    @test hist.v == [2.0]
    @test hist.policy[1][1] == [0.0, 1.0]
    @test hist.policy[2][1] == [0.0, 1.0]

    rollout_oracle = Fixtures.TableOracle(
        values=Dict(1f0 => 4f0, 2f0 => 0f0),
        policies=Dict(0f0 => (Float32[0.1, 0.9], Float32[0.9, 0.1])),
    )
    rollout_params = AZ.MCTSParams(
        tree_queries=0,
        c=0.5,
        max_depth=2,
        max_time=1.0,
        matrix_solver=Fixtures.GreedyMatrixSolver(),
        oracle=rollout_oracle,
        value_target=:rollout,
    )
    rollout_hist = AZ.mcts_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0)
    @test length(rollout_hist.v) == 1
    @test isapprox(rollout_hist.v[1], 4.0; atol=1e-6)

    bad_params = AZ.MCTSParams(
        tree_queries=0,
        matrix_solver=Fixtures.GreedyMatrixSolver(),
        oracle=oracle,
        value_target=:bad,
    )
    @test_throws ArgumentError AZ.mcts_sim(bad_params, game, false; progress=false, ϵ=0.0)
end
