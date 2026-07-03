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
end

@testset "MCTS regret transfer" begin
    game = Fixtures.ScalarMatrixGame()
    transfer_oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        regrets=Dict(0f0 => (Float32[0.8, -0.2], Float32[-0.1, 0.9])),
        strategies=Dict(0f0 => (Float32[0.25, 0.75], Float32[0.6, 0.4])),
    )
    τ = 1.5
    tw = 0.25

    # Warm-start injects the (unprojected) prior into the regret/policy_sum
    # accumulators, mirroring SM-OOS transfer.
    params = AZ.MCTSSearch(tree_queries=0, max_depth=2, τ=τ, transfer_weight=tw, oracle=transfer_oracle)
    @test AZ.has_regret_transfer(params)
    tree = AZ.Tree(params, game, false)
    AZ.expand_s!(tree, 1, game, transfer_oracle)
    AZ.warmstart_node!(params, tree, 1, game)
    regret_mass = sqrt(tw * τ)
    @test isapprox(tree.regret[1][1], regret_mass .* [0.8, -0.2]; atol=1e-6)
    @test isapprox(tree.regret[2][1], regret_mass .* [-0.1, 0.9]; atol=1e-6)
    @test isapprox(tree.policy_sum[1][1], τ .* [0.25, 0.75]; atol=1e-6)
    @test isapprox(tree.policy_sum[2][1], τ .* [0.6, 0.4]; atol=1e-6)

    # With no search iterations the targets decontaminate to zero regret and
    # uniform strategy: the oracle never trains on its own prior.
    yr, ys = AZ.mcts_root_targets(params, tree, game, 1)
    @test isapprox(yr[1], [0.0, 0.0]; atol=1e-9)
    @test isapprox(yr[2], [0.0, 0.0]; atol=1e-9)
    @test isapprox(ys[1], [0.5, 0.5]; atol=1e-6)
    @test isapprox(ys[2], [0.5, 0.5]; atol=1e-6)

    # Potential-bound projection clamps ||Q⁺||₂ ≤ sqrt(τ|A|)Δ, direction preserved.
    Δ = 0.1
    bounded = AZ.MCTSSearch(tree_queries=0, max_depth=2, τ=τ, transfer_weight=tw,
                            transfer_payoff_bound=Δ, oracle=transfer_oracle)
    btree = AZ.Tree(bounded, game, false)
    AZ.expand_s!(btree, 1, game, transfer_oracle)
    AZ.warmstart_node!(bounded, btree, 1, game)
    q1 = btree.regret[1][1]
    pot = sqrt(sum(x -> max(x, 0.0)^2, q1))
    @test isapprox(pot, sqrt(τ * 2) * Δ; atol=1e-9)
    @test isapprox(q1[1] / q1[2], 0.8 / -0.2; atol=1e-9)

    # No transfer configured ⇒ accumulators start at zero (original behavior).
    plain = AZ.MCTSSearch(tree_queries=0, max_depth=2, oracle=transfer_oracle)
    @test !AZ.has_regret_transfer(plain)
    ptree = AZ.Tree(plain, game, false)
    AZ.expand_s!(ptree, 1, game, transfer_oracle)
    AZ.warmstart_node!(plain, ptree, 1, game)
    @test all(iszero, ptree.regret[1][1])
    @test all(iszero, ptree.policy_sum[1][1])

    # End-to-end regret self-play emits regret/strategy targets of correct shape.
    Random.seed!(3)
    sim_params = AZ.MCTSSearch(tree_queries=32, max_depth=2, τ=τ, transfer_weight=tw, oracle=transfer_oracle)
    hist = AZ.mcts_regret_sim(sim_params, game, false; progress=false, ϵ=0.1, gae_lambda=1.0)
    @test haskey(hist, :regret) && haskey(hist, :strategy)
    @test length(hist.regret[1]) == length(hist.s) == length(hist.v)
    @test all(v -> length(v) == 2, hist.regret[1])
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[1][1])), 1.0; atol=1e-6)
end
