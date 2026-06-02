using Random

@testset "tree.jl and fitted SM-OOS" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        regrets=Dict(0f0 => (Float32[0.8, -0.2], Float32[-0.1, 0.9])),
        strategies=Dict(0f0 => (Float32[0.25, 0.75], Float32[0.6, 0.4])),
    )
    params = AZ.SMOOSParams(
        oos_iterations=0,
        τ=6.0,
        max_depth=2,
        oracle=oracle,
    )
    tree = AZ.Tree(params, game, false)
    @test fieldnames(typeof(tree)) == (:s, :children, :regret, :strategy)
    @test !(:n_s in fieldnames(typeof(tree)))
    @test !(:n_sa in fieldnames(typeof(tree)))

    AZ.expand_node!(tree, 1, game, params)
    @test isapprox(tree.regret[1][1], 6 .* [0.8, -0.2]; atol=1e-6)
    @test isapprox(tree.regret[2][1], 6 .* [-0.1, 0.9]; atol=1e-6)
    @test isapprox(tree.strategy[1][1], 6 .* [0.25, 0.75]; atol=1e-6)
    @test isapprox(tree.strategy[2][1], 6 .* [0.6, 0.4]; atol=1e-6)

    yr, ys = AZ.root_targets(params, tree, game, 1)
    @test isapprox(yr[1], [0.8, -0.2]; atol=1e-6)
    @test isapprox(yr[2], [-0.1, 0.9]; atol=1e-6)
    @test isapprox(ys[1], [0.25, 0.75]; atol=1e-6)
    @test isapprox(ys[2], [0.6, 0.4]; atol=1e-6)

    @test AZ.uniform(3) == fill(1 / 3, 3)
    @test AZ.eps_exploration([1.0, 0.0], 0.2) ≈ [0.9, 0.1]
    @test AZ.zs_reward_scalar((3.0, -3.0)) == 3.0
    @test AZ.zs_reward_scalar(2.5) == 2.5

    Random.seed!(7)
    idx = AZ.action_idx_from_probs([0.0, 1.0], [1.0, 0.0])
    @test idx == CartesianIndex(2, 1)

    search_params = AZ.SMOOSParams(
        oos_iterations=64,
        τ=0.0,
        max_depth=2,
        oracle=oracle,
    )
    (yr2, ys2), info = AZ.fitted_smoos_info(search_params, game, false; ϵ=0.0)
    @test length(yr2[1]) == 2
    @test length(yr2[2]) == 2
    @test length(ys2[1]) == 2
    @test length(ys2[2]) == 2
    @test all(isfinite, yr2[1])
    @test all(isfinite, yr2[2])
    @test sum(AZ.normalized_or_uniform(ys2[1])) ≈ 1.0
    @test sum(AZ.normalized_or_uniform(ys2[2])) ≈ 1.0
    @test info.tree isa AZ.SMOOSTree

    step_oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 0.5f0, 2f0 => 0f0),
        regrets=Dict(
            0f0 => (Float32[0.0, 0.0], Float32[0.0, 0.0]),
            1f0 => (Float32[0.0, 0.0], Float32[0.0, 0.0]),
        ),
        strategies=Dict(
            0f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
            1f0 => (Float32[0.5, 0.5], Float32[0.5, 0.5]),
        ),
    )
    Random.seed!(2)
    rollout_params = AZ.SMOOSParams(
        oos_iterations=1,
        τ=0.0,
        max_depth=3,
        oracle=step_oracle,
    )
    (_, _), rollout_info = AZ.fitted_smoos_info(rollout_params, Fixtures.TwoStepGame(), 0; ϵ=0.0)
    @test length(rollout_info.tree.s) == 3
    @test !isempty(rollout_info.tree.regret[1][1])
    @test !isempty(rollout_info.tree.regret[1][2])

    hist = AZ.smoos_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0, gae_lambda=1.0)
    @test length(hist.s) == 2
    @test length(hist.r) == 2
    @test length(hist.v) == 2
    @test length(hist.search_time) == 2
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[1][1])), 1.0; atol=1e-6)
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[2][1])), 1.0; atol=1e-6)
end
