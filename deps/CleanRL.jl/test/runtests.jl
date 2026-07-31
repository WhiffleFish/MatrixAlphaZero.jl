using Test

using CleanRL
using POMDPModels
using POMDPs

make_mdp() = SimpleGridWorld()

function finishes(f)
  f()
  true
end

@testset "MDPEnv smoke" begin
  mdp = make_mdp()
  env = MDPEnv(mdp; max_steps=1)

  @test state(env) isa Vector{Float32}
  @test state_dim(env) == 2
  @test action_count(env) == length(collect(POMDPs.actions(mdp)))
  @test random_action(env) in POMDPs.actions(mdp)
  @test !is_terminated(env)

  step!(env, 1)
  @test env.step_count == 1
  @test current_reward(env) isa Float32
  @test is_terminated(env)

  reset!(env)
  @test env.step_count == 0
  @test current_reward(env) == 0.0f0
  @test !is_terminated(env)
end

@testset "MultiThreadEnv smoke" begin
  env = MultiThreadEnv(() -> MDPEnv(make_mdp(); max_steps=1), 2)

  @test size(state(env)) == (2, 2)
  @test current_reward(env) == Float32[0, 0]
  @test is_terminated(env) == Bool[0, 0]

  step!(env, [1, 4])
  @test size(state(env)) == (2, 2)
  @test current_reward(env) isa Vector{Float32}
  @test is_terminated(env) == Bool[1, 1]

  reset!(env)
  @test size(state(env)) == (2, 2)
  @test is_terminated(env) == Bool[0, 0]
end

@testset "GAE" begin
  # γ=λ=0.5 and integer values/rewards keep every intermediate value an exact
  # binary fraction, so the expected advantages can be checked with `==`.
  γ = 0.5f0
  λ = 0.5f0

  @testset "no terminals" begin
    values = Float32[1, 1, 1, 1]      # [0, k], k=3
    rewards = Float32[2, 2, 2]        # [1, k]
    terminals = falses(4)             # [0, k]

    advantages = CleanRL.gae(values, rewards, terminals, γ, λ)

    @test length(advantages) == length(rewards)
    @test all(isfinite, advantages)
    # regression check for the bug where the loop skipped t == length(rewards),
    # leaving advantages[end] as whatever garbage `similar` handed back
    @test advantages[end] == 1.5f0
    @test advantages == Float32[1.96875, 1.875, 1.5]
  end

  @testset "terminal cuts bootstrap" begin
    values = Float32[1, 1, 1, 1]
    rewards = Float32[2, 2, 2]
    terminals = Bool[0, 0, 1, 0]       # episode ends after step 2

    advantages = CleanRL.gae(values, rewards, terminals, γ, λ)

    @test length(advantages) == length(rewards)
    @test all(isfinite, advantages)
    @test advantages[end] == 1.5f0
    @test advantages == Float32[1.75, 1.0, 1.5]
  end
end

@testset "Algorithm smoke" begin
  mdp = make_mdp()

  @test finishes(() -> dqn(mdp, DQNConfig(
    run_name="test-dqn",
    total_timesteps=3,
    min_buff_size=10,
  ); max_steps=1))

  @test finishes(() -> a2c(mdp, A2CConfig(
    run_name="test-a2c",
    total_timesteps=3,
    min_replay_size=10,
  ); max_steps=1))

  @test finishes(() -> ppo(mdp, PPOConfig(
    total_timesteps=8,
    num_steps=2,
    num_envs=2,
    num_minibatches=1,
    update_epochs=1,
    anneal_lr=true,
  ); max_steps=1))
end
