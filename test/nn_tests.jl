using Flux

@testset "nn.jl" begin
    model = Fixtures.simple_fitted_regret_model(input_dim=2)
    weighted_model = Fixtures.simple_fitted_regret_model(;
        input_dim=2,
        value_weight=0.25f0,
        regret_weight=0.5f0,
        strategy_weight=0.75f0,
    )
    @test weighted_model.value_weight == 0.25f0
    @test weighted_model.regret_weight == 0.5f0
    @test weighted_model.strategy_weight == 0.75f0
    @test all(x -> x isa AbstractArray, Flux.trainables(weighted_model))
    x = Float32[
        1 0
        0 1
    ]
    value_target = Float32[0.25, -0.5]
    regret_target = (
        Float32[
            0.5 -0.5
            -0.2 0.2
        ],
        Float32[
            -0.1 0.1
            0.3 -0.3
        ],
    )
    strategy_target = (
        Float32[
            1 0
            0 1
        ],
        Float32[
            0 1
            1 0
        ],
    )

    out = model(x)
    @test size(out.value) == (1, 2)
    @test length(out.regret) == 2
    @test length(out.strategy) == 2
    @test all(isapprox.(sum(out.strategy[1]; dims=1), 1.0f0; atol=1e-6))
    @test all(isapprox.(sum(out.strategy[2]; dims=1), 1.0f0; atol=1e-6))

    lv, lr, ls = AZ.loss(model, x, value_target, regret_target, strategy_target)
    @test lv > 0
    @test lr > 0
    @test ls > 0
    @test size(AZ.value(model, x)) == (1, 2)
    @test length(AZ.regret(model, x)) == 2
    @test length(AZ.strategy(model, x)) == 2

    loss_info = AZ.getloss(model, x; value_target, regret_target, strategy_target)
    @test hasproperty(loss_info, :value_loss)
    @test hasproperty(loss_info, :value_mse)
    @test hasproperty(loss_info, :regret_loss)
    @test hasproperty(loss_info, :strategy_loss)

    ac = Fixtures.simple_actor_critic(; input_dim=2, value_weight=0.4f0, policy_weight=0.6f0)
    ac_out = ac(x)
    @test ac.value_weight == 0.4f0
    @test ac.policy_weight == 0.6f0
    @test all(x -> x isa AbstractArray, Flux.trainables(ac))
    @test size(ac_out.value) == (1, 2)
    @test length(ac_out.policy) == 2
    @test all(isapprox.(sum(ac_out.policy[1]; dims=1), 1.0f0; atol=1e-6))
    @test all(isapprox.(sum(ac_out.policy[2]; dims=1), 1.0f0; atol=1e-6))
    lv_ac, lp_ac = AZ.loss(ac, x, value_target, strategy_target)
    @test lv_ac > 0
    @test lp_ac > 0
    @test size(AZ.value(ac, x)) == (1, 2)
    @test length(AZ.policy(ac, x)) == 2
    ac_loss_info = AZ.getloss(ac, x; value_target, policy_target=strategy_target)
    @test hasproperty(ac_loss_info, :value_loss)
    @test hasproperty(ac_loss_info, :value_mse)
    @test hasproperty(ac_loss_info, :policy_loss)

    head = AZ.MultiActor(Dense(2 => 2), Dense(2 => 3))
    head_out = head(Float32[1, 2])
    @test length(head_out) == 2
    @test isapprox(sum(head_out[1]), 1.0f0; atol=1e-6)
    @test isapprox(sum(head_out[2]), 1.0f0; atol=1e-6)
    @test AZ.fitted_strategy_loss(head, Float32[1 0; 0 1], (
        Float32[
            1 0
            0 1
        ],
        Float32[
            1 0
            0 0
            0 1
        ],
    )) > 0

    critic = AZ.HLGaussCritic(Dense(2 => 6), -1f0, 1f0, 6)
    probs = AZ.transform_to_probs(critic, 0.2)
    @test length(probs) == 6
    @test isapprox(sum(probs), 1.0f0; atol=1e-5)
    @test isapprox(AZ.transform_from_probs(critic, probs), 0.2f0; atol=0.2)

    batch_probs = AZ.prepare_target(critic, Float32[-2, 0, 2])
    @test size(batch_probs) == (6, 3)
    @test all(isapprox.(sum(batch_probs; dims=1), 1.0f0; atol=1e-5))
    @test AZ.criticloss(critic, rand(Float32, 2, 3), batch_probs) > 0
    @test AZ.prepare_target(model.critic, value_target) === value_target

    static_model = AZ.StaticFittedRegretModel(
        s -> (Float32[0.5, -0.5], Float32[-0.25, 0.25]),
        s -> (Float32[0.75, 0.25], Float32[0.4, 0.6]),
        s -> Float32(sum(s)),
    )
    states = [Float32[1, 2], Float32[3, 4]]
    @test AZ.state_value(static_model, nothing, states[1]) == 3f0
    @test AZ.batch_state_value(static_model, nothing, states) == Float32[3, 7]
    @test AZ.state_regret(static_model, nothing, states[1])[1] == Float32[0.5, -0.5]
    @test AZ.state_strategy(static_model, nothing, states[1]) == (Float32[0.75, 0.25], Float32[0.4, 0.6])

    static_ac = AZ.StaticActorCritic(
        s -> (Float32[0.2, 0.8], Float32[0.6, 0.4]),
        s -> Float32(sum(s));
        value_weight=0.2f0,
        policy_weight=0.3f0,
    )
    @test static_ac.value_weight == 0.2f0
    @test static_ac.policy_weight == 0.3f0
    @test AZ.state_value(static_ac, nothing, states[1]) == 3f0
    @test AZ.batch_state_value(static_ac, nothing, states) == Float32[3, 7]
    @test AZ.state_policy(static_ac, nothing, states[1]) == (Float32[0.2, 0.8], Float32[0.6, 0.4])

    game = Fixtures.TwoStepGame()
    model1 = Fixtures.simple_fitted_regret_model()
    @test size(AZ.state_value(model1, game, 1)) == (1,)
    @test length(AZ.state_regret(model1, game, 1)) == 2
    @test length(AZ.state_strategy(model1, game, 1)) == 2
    @test size(AZ.batch_state_value(model1, game, [0, 1])) == (1, 2)
    @test length(AZ.batch_state_regret(model1, game, [0, 1])) == 2
    @test length(AZ.batch_state_strategy(model1, game, [0, 1])) == 2

    ac1 = Fixtures.simple_actor_critic()
    @test size(AZ.state_value(ac1, game, 1)) == (1,)
    @test length(AZ.state_policy(ac1, game, 1)) == 2
    @test size(AZ.batch_state_value(ac1, game, [0, 1])) == (1, 2)
    @test length(AZ.batch_state_policy(ac1, game, [0, 1])) == 2
end
