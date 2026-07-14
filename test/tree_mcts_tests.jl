using Random

@testset "tree.jl and fitted SM-OOS" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.TableOracle(
        values=Dict(0f0 => 0f0, 1f0 => 5f0),
        regrets=Dict(0f0 => (Float32[0.8, -0.2], Float32[-0.1, 0.9])),
        strategies=Dict(0f0 => (Float32[0.25, 0.75], Float32[0.6, 0.4])),
    )
    previous_mass = 6.0
    transfer_weight = 0.25
    retained_mass = transfer_weight * previous_mass
    params = AZ.SMOOSSearch(
        oos_iterations=0,
        τ=retained_mass,
        transfer_weight=transfer_weight,
        max_depth=2,
        oracle=oracle,
    )
    tree = AZ.Tree(params, game, false)
    @test fieldnames(typeof(tree)) == (:s, :children, :prior, :regret, :strategy)
    @test !(:n_s in fieldnames(typeof(tree)))
    @test !(:n_sa in fieldnames(typeof(tree)))

    AZ.expand_node!(tree, 1, game, params)
    regret_mass = transfer_weight * sqrt(previous_mass)
    @test regret_mass ≈ sqrt(transfer_weight * retained_mass)
    @test isapprox(tree.regret[1][1], regret_mass .* [0.8, -0.2]; atol=1e-6)
    @test isapprox(tree.regret[2][1], regret_mass .* [-0.1, 0.9]; atol=1e-6)
    @test isapprox(tree.strategy[1][1], retained_mass .* [0.25, 0.75]; atol=1e-6)
    @test isapprox(tree.strategy[2][1], retained_mass .* [0.6, 0.4]; atol=1e-6)
    @test tree.prior[1][1] ≈ [0.25, 0.75]
    @test tree.prior[2][1] ≈ [0.6, 0.4]

    adaptive = AZ.SMOOSSearch(
        oos_iterations=4,
        τ=10.0,
        loss_scaled_transfer=AZ.LossScaledTransfer(
            regret_scale=0.5,
            strategy_scale=2.0,
            reach_power=1.0,
        ),
        regret_confidence=1.0,
        strategy_confidence=0.5,
        max_depth=2,
        oracle=oracle,
    )
    @test collect(AZ.transfer_pseudo_masses(adaptive, 1.0)) ≈ [2.0, 4.0]
    @test collect(AZ.transfer_pseudo_masses(adaptive, 0.25)) ≈ [0.5, 1.0]
    (adaptive_regret, adaptive_strategy) = AZ.transfer_prior(adaptive, game, false, 0.25)
    @test adaptive_regret[1] ≈ sqrt(0.5) .* [0.8, -0.2]
    @test adaptive_strategy[1] ≈ [0.25, 0.75]

    # Targets are decontaminated: with zero search iterations the prior
    # initialization is subtracted back out, leaving no self-distillation.
    yr, ys = AZ.root_targets(params, tree, game, 1)
    @test isapprox(yr[1], [0.0, 0.0]; atol=1e-9)
    @test isapprox(yr[2], [0.0, 0.0]; atol=1e-9)
    @test isapprox(ys[1], [0.5, 0.5]; atol=1e-6)
    @test isapprox(ys[2], [0.5, 0.5]; atol=1e-6)

    # Potential-bound projection: with a finite payoff bound Δ the transferred
    # regret initialization satisfies ||Q⁺||₂ ≤ sqrt(τ|A|)Δ, direction preserved.
    Δ = 0.1
    bounded_params = AZ.SMOOSSearch(
        oos_iterations=0,
        τ=retained_mass,
        transfer_weight=transfer_weight,
        transfer_payoff_bound=Δ,
        max_depth=2,
        oracle=oracle,
    )
    bounded_tree = AZ.Tree(bounded_params, game, false)
    AZ.expand_node!(bounded_tree, 1, game, bounded_params)
    q1 = bounded_tree.regret[1][1]
    pot = sqrt(sum(x -> max(x, 0.0)^2, q1))
    bound = sqrt(retained_mass * 2) * Δ
    @test pot ≤ bound + 1e-9
    @test isapprox(pot, bound; atol=1e-9)  # unprojected mass exceeds bound, so it lands on it
    @test isapprox(q1[1] / q1[2], 0.8 / -0.2; atol=1e-9)
    # strategy initialization is unaffected by the regret projection
    @test isapprox(bounded_tree.strategy[1][1], retained_mass .* [0.25, 0.75]; atol=1e-6)
    # decontamination subtracts the projected init exactly
    yr_b, ys_b = AZ.root_targets(bounded_params, bounded_tree, game, 1)
    @test isapprox(yr_b[1], [0.0, 0.0]; atol=1e-9)
    @test isapprox(ys_b[1], [0.5, 0.5]; atol=1e-6)

    @test AZ.uniform(3) == fill(1 / 3, 3)
    @test AZ.eps_exploration([1.0, 0.0], 0.2) ≈ [0.9, 0.1]
    @test AZ.zs_reward_scalar((3.0, -3.0)) == 3.0
    @test AZ.zs_reward_scalar(2.5) == 2.5

    Random.seed!(7)
    idx = AZ.action_idx_from_probs([0.0, 1.0], [1.0, 0.0])
    @test idx == CartesianIndex(2, 1)

    deep_search = AZ.SMOOSSearch(
        oos_iterations=64,
        τ=0.0,
        max_depth=2,
        oracle=oracle,
    )
    (yr2, ys2), info = AZ.fitted_smoos_info(deep_search, game, false; ϵ=0.0)
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
    rollout_params = AZ.SMOOSSearch(
        oos_iterations=1,
        τ=0.0,
        max_depth=3,
        oracle=step_oracle,
    )
    (_, _), rollout_info = AZ.fitted_smoos_info(rollout_params, Fixtures.TwoStepGame(), 0; ϵ=0.0)
    @test length(rollout_info.tree.s) == 3
    @test !isempty(rollout_info.tree.regret[1][1])
    @test !isempty(rollout_info.tree.regret[1][2])
    @test sum(rollout_info.tree.strategy[1][1]) > 0
    @test sum(rollout_info.tree.strategy[2][1]) > 0

    hist = AZ.smoos_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0, gae_lambda=1.0)
    @test length(hist.s) == 2
    @test length(hist.r) == 2
    @test length(hist.v) == 2
    @test length(hist.search_time) == 2
    shallow_hist = AZ.smoos_sim(rollout_params, Fixtures.TwoStepGame(), 0; progress=false, ϵ=0.0, sim_depth=1, gae_lambda=1.0)
    @test length(shallow_hist.s) == 1
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[1][1])), 1.0; atol=1e-6)
    @test isapprox(sum(AZ.normalized_or_uniform(hist.strategy[2][1])), 1.0; atol=1e-6)
end
