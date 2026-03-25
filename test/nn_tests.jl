using Flux

@testset "nn.jl" begin
    ac = Fixtures.simple_actor_critic(input_dim=2)
    x = Float32[
        1 0
        0 1
    ]
    value_target = Float32[0.25, -0.5]
    policy_target = (
        Float32[
            1 0
            0 1
        ],
        Float32[
            0 1
            1 0
        ],
    )

    out = ac(x)
    @test size(out.value) == (1, 2)
    @test length(out.policy) == 2
    @test all(isapprox.(sum(out.policy[1]; dims=1), 1.0f0; atol=1e-6))
    @test all(isapprox.(sum(out.policy[2]; dims=1), 1.0f0; atol=1e-6))

    lv, lp = AZ.loss(ac, x, value_target, policy_target)
    @test lv > 0
    @test lp > 0
    @test size(AZ.value(ac, x)) == (1, 2)
    @test length(AZ.policy(ac, x)) == 2

    loss_info = AZ.getloss(ac, x; value_target, policy_target)
    @test hasproperty(loss_info, :value_loss)
    @test hasproperty(loss_info, :value_mse)
    @test hasproperty(loss_info, :policy_loss)

    actor = AZ.MultiActor(Dense(2 => 2), Dense(2 => 3))
    actor_out = actor(Float32[1, 2])
    @test length(actor_out) == 2
    @test isapprox(sum(actor_out[1]), 1.0f0; atol=1e-6)
    @test isapprox(sum(actor_out[2]), 1.0f0; atol=1e-6)
    @test AZ.actorloss(actor, Float32[1 0; 0 1], (
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
    @test AZ.prepare_target(ac.critic, value_target) === value_target

    static_ac = AZ.StaticActorCritic(
        s -> (Float32[0.75, 0.25], Float32[0.4, 0.6]),
        s -> Float32(sum(s)),
    )
    states = [Float32[1, 2], Float32[3, 4]]
    @test AZ.state_value(static_ac, nothing, states[1]) == 3f0
    @test AZ.batch_state_value(static_ac, nothing, states) == Float32[3, 7]
    @test AZ.state_policy(static_ac, nothing, states[1]) == (Float32[0.75, 0.25], Float32[0.4, 0.6])
    @test AZ.batch_state_policy(static_ac, nothing, states)[1][1] == Float32[0.75, 0.25]

    game = Fixtures.TwoStepGame()
    ac1 = Fixtures.simple_actor_critic()
    @test size(AZ.state_value(ac1, game, 1)) == (1,)
    @test length(AZ.state_policy(ac1, game, 1)) == 2
    @test size(AZ.batch_state_value(ac1, game, [0, 1])) == (1, 2)
    @test length(AZ.batch_state_policy(ac1, game, [0, 1])) == 2
end
