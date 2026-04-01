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
        max_depth=2,
        max_time=1.0,
        search_style=AZ.MatrixGameSearch(c=0.5, matrix_solver=Fixtures.GreedyMatrixSolver()),
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
        max_depth=2,
        max_time=1.0,
        search_style=AZ.MatrixGameSearch(c=0.5, matrix_solver=Fixtures.GreedyMatrixSolver()),
        oracle=rollout_oracle,
        value_target=:rollout,
    )
    rollout_hist = AZ.mcts_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0)
    @test length(rollout_hist.v) == 1
    @test isapprox(rollout_hist.v[1], 4.0; atol=1e-6)

    bad_params = AZ.MCTSParams(
        tree_queries=0,
        search_style=AZ.MatrixGameSearch(matrix_solver=Fixtures.GreedyMatrixSolver()),
        oracle=oracle,
        value_target=:bad,
    )
    @test_throws ArgumentError AZ.mcts_sim(bad_params, game, false; progress=false, ϵ=0.0)

    bandit_oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 0f0),
        policies=Dict(0f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5])),
    )

    Random.seed!(1)
    rm_params = AZ.MCTSParams(
        tree_queries=400,
        max_depth=4,
        max_time=1.0,
        oracle=bandit_oracle,
        search_style=AZ.RegretMatchingSearch(),
    )
    (x_rm, y_rm, v_rm), rm_info = AZ.search_info(rm_params, game, false; ϵ=0.1)
    @test x_rm[2] > x_rm[1]
    @test y_rm[1] > y_rm[2]
    @test v_rm > 0.6
    @test rm_info.tree.return_sum[1] > 0.0

    Random.seed!(1)
    exp3_params = AZ.MCTSParams(
        tree_queries=400,
        max_depth=4,
        max_time=1.0,
        oracle=bandit_oracle,
        search_style=AZ.Exp3Search(),
    )
    (x_exp3, y_exp3, v_exp3), exp3_info = AZ.search_info(exp3_params, game, false; ϵ=0.1)
    @test x_exp3[2] > x_exp3[1]
    @test y_exp3[1] > y_exp3[2]
    @test v_exp3 > 0.6
    @test exp3_info.tree.return_sum[1] > 0.0

    step_oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 0.5f0, 2f0 => 0f0),
        policies=Dict(
            0f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
            1f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
        ),
    )
    Random.seed!(2)
    rm_mean_params = AZ.MCTSParams(
        tree_queries=500,
        max_depth=3,
        max_time=1.0,
        oracle=step_oracle,
        search_style=AZ.RegretMatchingSearch(backup=:mean),
    )
    (x_rm_mean, y_rm_mean, v_rm_mean), rm_mean_info = AZ.search_info(rm_mean_params, Fixtures.TwoStepGame(), 0; ϵ=0.1)
    @test x_rm_mean[1] > 0.9
    @test y_rm_mean[1] > 0.9
    @test isapprox(v_rm_mean, 1.25; atol=0.1)
    @test isapprox(v_rm_mean, rm_mean_info.tree.return_sum[1] / rm_mean_info.tree.n_s[1]; atol=1e-6)
end
