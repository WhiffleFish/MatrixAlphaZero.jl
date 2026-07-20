using ProgressMeter
using POMDPs
using Random

@testset "MCTS search" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.static_actor_critic(
        game;
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        policies=Dict(0f0 => (Float32[0.8, 0.2], Float32[0.1, 0.9])),
    )
    params = AZ.MCTSSearch(
        tree_queries=400,
        max_depth=2,
        max_time=1.0,
        oracle=oracle,
    )

    tree = AZ.Tree(params, game, false)
    @test tree isa AZ.SearchTree
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
    @test_throws MethodError AZ.RegretMatchingSearch(target_policy=:current)
    @test AZ.RegretMatchingSearch().method isa AZ.Vanilla
    @test AZ.RegretMatchingSearch(:mean).backup == :mean
    @test AZ.RegretMatchingSearch(method=AZ.Plus()).method isa AZ.Plus

    vanilla_regret = [-1.0, 0.5]
    plus_regret = copy(vanilla_regret)
    delta = [0.25, -1.0]
    AZ.accumulate_regret!(AZ.Vanilla(), vanilla_regret, delta)
    AZ.accumulate_regret!(AZ.Plus(), plus_regret, delta)
    @test vanilla_regret == [-0.75, -0.5]
    @test plus_regret == [0.0, 0.0]

    Random.seed!(7)
    idx = AZ.action_idx_from_probs([0.0, 1.0], [1.0, 0.0])
    @test idx == CartesianIndex(2, 1)

    (x, y, v), info = AZ.search_info(params, game, false; ϵ=0.0)
    @test x[2] > x[1]
    @test y[1] > y[2]
    @test v > 0.0
    @test info.tree isa AZ.SearchTree
    @test length(info.tree.s) == 5

    x2, y2, v2 = AZ.search(params, game, false; ϵ=0.0)
    @test x2[2] > x2[1]
    @test y2[1] > y2[2]
    @test v2 > 0.0

    tree2 = AZ.Tree(params, game, false)
    @test AZ.simulate(params, tree2, game, 1; ϵ=0.0) == 0.0

    # Epsilon changes traversal only. The average-strategy accumulator must
    # contain the unperturbed regret-matching policy so self-play does not
    # apply exploration twice or train the strategy head on epsilon noise.
    exploration_tree = AZ.Tree(params, game, false)
    AZ.simulate(params, exploration_tree, game, 1; ϵ=1.0) # expand root
    exploration_tree.regret[1][1] .= [1.0, 0.0]
    exploration_tree.regret[2][1] .= [0.0, 1.0]
    @test AZ.current_policy(params.search_style, exploration_tree, 1) == ([1.0, 0.0], [0.0, 1.0])
    @test AZ.selection_policy(params.search_style, exploration_tree, 1; ϵ=1.0) == ([0.5, 0.5], [0.5, 0.5])
    AZ.simulate(params, exploration_tree, game, 1; ϵ=1.0)
    @test exploration_tree.policy_sum[1][1] == [1.0, 0.0]
    @test exploration_tree.policy_sum[2][1] == [0.0, 1.0]

    hist = AZ.mcts_sim(params, game, false; progress=false, ϵ=0.0)
    @test length(hist.s) == 1
    @test only(hist.r) ∈ vec(game.rewards)
    @test only(hist.v) > 0.0
    @test length(hist.search_time) == 1
    @test only(hist.search_time) >= 0.0
    @test isapprox(sum(hist.policy[1][1]), 1.0; atol=1e-6)
    @test isapprox(sum(hist.policy[2][1]), 1.0; atol=1e-6)

    progress = Progress(1; enabled=false)
    serial = AZ.serial_mcts(progress, game, params, 1, initialstate(game); ϵ=0.0, sim_depth=1)
    @test length(serial) == 1
    @test hasproperty(only(serial), :policy)

    rollout_oracle = Fixtures.static_actor_critic(
        Fixtures.TwoStepGame();
        values=Dict(1f0 => 4f0, 2f0 => 0f0),
        policies=Dict(0f0 => (Float32[0.1, 0.9], Float32[0.9, 0.1])),
    )
    rollout_params = AZ.MCTSSearch(
        tree_queries=0,
        max_depth=2,
        max_time=1.0,
        oracle=rollout_oracle,
        value_target=:rollout,
    )
    rollout_hist = AZ.mcts_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0, sim_depth=1)
    @test length(rollout_hist.v) == 1
    @test length(rollout_hist.search_time) == 1
    @test isapprox(rollout_hist.v[1], 2.0; atol=1e-6)

    bad_params = AZ.MCTSSearch(tree_queries=0, oracle=oracle, value_target=:bad)
    @test_throws ArgumentError AZ.mcts_sim(bad_params, game, false; progress=false, ϵ=0.0)

    step_oracle = Fixtures.static_actor_critic(
        Fixtures.TwoStepGame();
        values=Dict(0f0 => 0f0, 1f0 => 0.5f0, 2f0 => 0f0),
        policies=Dict(
            0f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
            1f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
        ),
    )
    Random.seed!(2)
    rm_mean_params = AZ.MCTSSearch(
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

    plus_params = AZ.MCTSSearch(
        tree_queries=500,
        max_depth=3,
        max_time=1.0,
        oracle=step_oracle,
        search_style=AZ.RegretMatchingSearch(method=AZ.Plus()),
    )
    (_, _, _), plus_info = AZ.search_info(plus_params, Fixtures.TwoStepGame(), 0; ϵ=0.1)
    @test all(all(≥(0.0), regret) for player in plus_info.tree.regret for regret in player)
    @test all(all(≥(0.0), regret) for player in plus_info.tree.fresh_regret for regret in player)
end

@testset "MCTS inference priors and regret training targets" begin
    game = Fixtures.ScalarMatrixGame()
    transfer_oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        regrets=Dict(0f0 => (Float32[0.8, -0.2], Float32[-0.1, 0.9])),
        strategies=Dict(0f0 => (Float32[0.25, 0.75], Float32[0.6, 0.4])),
    )
    prior_scale = 4.0

    params = AZ.MCTSSearch(; tree_queries=0, max_depth=2, prior_scale, oracle=transfer_oracle)
    @test AZ.has_prior_transfer(params)
    tree = AZ.Tree(params, game, false)
    AZ.expand_s!(tree, 1, game, transfer_oracle)
    AZ.warmstart_node!(params, tree, 1, game)
    @test isapprox(tree.regret[1][1], prior_scale .* [0.8, -0.2]; atol=1e-6)
    @test isapprox(tree.regret[2][1], prior_scale .* [-0.1, 0.9]; atol=1e-6)
    @test isapprox(tree.policy_sum[1][1], prior_scale .* [0.25, 0.75]; atol=1e-6)
    @test isapprox(tree.policy_sum[2][1], prior_scale .* [0.6, 0.4]; atol=1e-6)

    # Deeper nodes attenuate the single fitted-prior scale by joint reach under
    # the learned average policy.
    (reached_regret, reached_strategy) = AZ.mcts_prior(params, game, false, 0.25)
    @test reached_regret[1] ≈ (prior_scale * 0.25) .* [0.8, -0.2]
    @test reached_strategy[1] ≈ (prior_scale * 0.25) .* [0.25, 0.75]

    plus_params = AZ.MCTSSearch(
        tree_queries=0,
        max_depth=2,
        prior_scale=prior_scale,
        oracle=transfer_oracle,
        search_style=AZ.RegretMatchingSearch(method=AZ.Plus()),
    )
    plus_tree = AZ.Tree(plus_params, game, false)
    AZ.expand_s!(plus_tree, 1, game, transfer_oracle)
    AZ.warmstart_node!(plus_params, plus_tree, 1, game)
    @test isapprox(plus_tree.regret[1][1], prior_scale .* [0.8, 0.0]; atol=1e-6)
    @test isapprox(plus_tree.regret[2][1], prior_scale .* [0.0, 0.9]; atol=1e-6)
    @test all(iszero, plus_tree.fresh_regret[1][1])

    # Targets exclude inference priors. Training additionally rejects nonzero
    # prior_scale at the solver boundary.
    yr, ys = AZ.mcts_root_targets(params, tree, game, 1)
    @test isapprox(yr[1], [0.0, 0.0]; atol=1e-9)
    @test isapprox(yr[2], [0.0, 0.0]; atol=1e-9)
    @test isapprox(ys[1], [0.5, 0.5]; atol=1e-6)
    @test isapprox(ys[2], [0.5, 0.5]; atol=1e-6)

    # No inference prior configured means every local solve starts from zero.
    plain = AZ.MCTSSearch(tree_queries=0, max_depth=2, oracle=transfer_oracle)
    @test !AZ.has_prior_transfer(plain)
    ptree = AZ.Tree(plain, game, false)
    AZ.expand_s!(ptree, 1, game, transfer_oracle)
    AZ.warmstart_node!(plain, ptree, 1, game)
    @test all(iszero, ptree.regret[1][1])
    @test all(iszero, ptree.policy_sum[1][1])

    # The regret network is fitted to average regret. A deployment scale of T
    # therefore recovers the cumulative-regret initialization from the theory.
    ptree.n_s[1] = 5 # one expansion query plus four local RM updates
    ptree.n_sa[1] .= [1 1; 1 1]
    ptree.fresh_regret[1][1] .= [4.0, -2.0]
    ptree.fresh_regret[2][1] .= [-1.0, 3.0]
    average_regret, _ = AZ.mcts_root_targets(plain, ptree, game, 1)
    @test average_regret[1] == [1.0, -0.5]
    @test average_regret[2] == [-0.25, 0.75]

    invalid = AZ.MCTSSearch(; tree_queries=0, prior_scale=-1.0, oracle=transfer_oracle)
    @test_throws ArgumentError AZ.has_prior_transfer(invalid)

    # End-to-end regret self-play emits regret/strategy targets of correct shape
    # and honors the configured value-supervision source.
    Random.seed!(3)
    sim_params = AZ.MCTSSearch(tree_queries=32, max_depth=2, oracle=transfer_oracle)
    hist = AZ.mcts_regret_sim(sim_params, game, false; progress=false, ϵ=0.1, gae_lambda=1.0)
    @test haskey(hist, :regret) && haskey(hist, :strategy)
    @test length(hist.regret[1]) == length(hist.s) == length(hist.v)
    @test all(v -> length(v) == 2, hist.regret[1])
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[1][1])), 1.0; atol=1e-6)

    Random.seed!(11)
    (_, _, expected_search_value), _ = AZ.search_info(sim_params, game, false; ϵ=0.1)
    Random.seed!(11)
    search_hist = AZ.mcts_regret_sim(
        sim_params,
        game,
        false;
        progress=false,
        ϵ=0.1,
        sim_depth=1,
        gae_lambda=1.0,
    )
    @test only(search_hist.v) ≈ expected_search_value

    gae_params = AZ.with_oracle(sim_params, transfer_oracle; value_target=:gae)
    Random.seed!(11)
    gae_hist = AZ.mcts_regret_sim(
        gae_params,
        game,
        false;
        progress=false,
        ϵ=0.1,
        sim_depth=1,
        gae_lambda=1.0,
    )
    @test only(gae_hist.v) ≈ only(gae_hist.r)

    bad_value_params = AZ.with_oracle(sim_params, transfer_oracle; value_target=:rollout)
    @test_throws ArgumentError AZ.mcts_regret_sim(
        bad_value_params,
        game,
        false;
        progress=false,
        ϵ=0.1,
    )
end
