@kwdef struct PPOConfig
  total_timesteps::Int = 500_000
  num_steps::Int = 32
  num_envs::Int = 4
  num_minibatches::Int = 4
  update_epochs::Int = 4

  lr::Float32 = 2.5f-4
  gamma::Float32 = 0.99
  gae_lambda::Float32 = 0.95

  clip_coef::Float32 = 0.2
  ent_coeff::Float32 = 0.01
  v_coef::Float32 = 0.5

  normalize_advantages::Bool = true
  clip_value_loss::Bool = true
  anneal_lr::Bool = true

  name::String = "ppo-test"
  log_dir::String = "logs"
end

function get_action(obs::AbstractVecOrMat{Float32}, actor::Chain)
  logits = actor(obs)
  probs = softmax(logits)
  logprobs = logsoftmax(logits)

  action = sample.(Weights.(eachcol(probs)))
  batch_size = last(size(obs))  # batch is last dim
  action_ind = CartesianIndex.(action, 1:batch_size)  # for 2D indexing
  logprob_action = logprobs[action_ind]

  action, logprob_action
end

function logprob_actions(obs::AbstractVecOrMat{Float32}, actor::Chain, actions::AbstractVector{<:Integer})
  logits = actor(obs)
  probs = softmax(logits)
  logprobs = logsoftmax(logits)

  batch_size = last(size(obs))  # batch is last dim
  action_ind = CartesianIndex.(actions, 1:batch_size)  # for 2D indexing
  logprob_action = logprobs[action_ind]
  entropy = vec(-sum(probs .* logprobs; dims=1))

  logprob_action, entropy
end


function gae(values::AbstractVector{T}, rewards::AbstractVector{T}, terminals::AbstractVector{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
  """
  Generalized advantage estimation.

  Args:
    values: [0, k]
    rewards: [1, k]
    terminals: [0,k]
    γ: gamma/discount
    λ: gae lambda

  Returns: 
   advatages [0, k-1]
  """
  advantages = similar(rewards)
  nonterm = 1.0 .- terminals

  lastgaelam = zero(T)
  for t in length(rewards):-1:1
    δ = rewards[t] + γ * nonterm[t+1] * values[t+1] - values[t]
    lastgaelam = δ + γ * λ * nonterm[t+1] * lastgaelam
    advantages[t] = lastgaelam
  end

  advantages
end

function ppo(mdp::POMDPs.MDP, config::PPOConfig=PPOConfig(); kwargs...)
  ppo(config) do
    MDPEnv(mdp; kwargs...)
  end
end

function ppo(env_factory::Function, config::PPOConfig=PPOConfig())
  nt = config.num_envs
  Logger.make_logger(config.name; log_dir=config.log_dir, to_terminal=false)

  env = MultiThreadEnv(env_factory, nt)

  actor, critic = Networks.make_actor_critic(single_action_count(env), single_state_dim(env)) .|> Flux.f32

  batch_size = config.num_steps * nt
  minibatch_size = batch_size ÷ config.num_minibatches
  num_updates = config.total_timesteps ÷ batch_size

  opt = Flux.OptimiserChain(ClipNorm(0.5), Adam(config.lr))  # one opt per network?
  opt_state = Flux.setup(opt, (actor, critic))

  transition = (
    state=rand(Float32, single_state_dim(env), nt),
    action=rand(1:single_action_count(env), nt),
    logprob=Float32.(ones(nt)),
    reward=Float32.(ones(nt)),
    terminal=fill(true, nt),
    value=Float32.(ones(nt)),
  )

  rb = Buffer.ReplayBuffer(transition, config.num_steps)

  global_step = 0
  last_log_step = 0
  episode_returns = zeros(nt)
  episode_lengths = zeros(nt)

  start_time = time()
  reset!(env; is_force=true)

  next_obs = state(env)
  next_done = is_terminated(env)

  for update in 1:num_updates
    if config.anneal_lr
      frac = 1.0 - (update - 1.0) / num_updates
      Flux.Optimisers.adjust!(opt_state, Float32(frac * config.lr))
    end

    for step in 1:config.num_steps
      global_step += nt
      episode_lengths .+= 1

      action, log_prob = get_action(next_obs, actor)
      value = critic(next_obs)

      step!(env, action)

      rewards = current_reward(env)
      Buffer.add!(rb, (
        state=next_obs,
        action=action,
        logprob=log_prob,
        reward=rewards,
        terminal=next_done,
        value=value,
      ))

      # todo: I don't like next obs - put this above
      next_obs = deepcopy(state(env))
      next_done = is_terminated(env)
      episode_returns += rewards

      if any(next_done)
        steps_per_sec = trunc(global_step / (time() - start_time))
        for i in 1:nt
          !next_done[i] && continue  # only log if terminal

          episode_return = episode_returns[i]
          episode_length = episode_lengths[i]

          # todo: would be nice if we could pass step instead of log_step_increment
          log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
          @info "Episode Statistics" episode_return episode_length global_step steps_per_sec log_step_increment = log_step_inc

          episode_lengths[i] = 0
          episode_returns[i] = 0
          last_log_step = deepcopy(global_step)
        end

        reset!(env)
        next_obs = deepcopy(state(env))
      end
    end

    # bootstrap value if not done
    next_obs = state(env)
    next_done = is_terminated(env)
    next_values = critic(next_obs)

    advantages = gae.(
      eachrow(hcat(rb.data.value, next_values')),
      eachrow(rb.data.reward),
      eachrow(hcat(rb.data.terminal, next_done)),
      config.gamma,
      config.gae_lambda
    )
    advantages = reduce(hcat, advantages)'  # stack
    returns = advantages + rb.data.value

    # flatten everything
    states = reshape(rb.data.state, :, batch_size)
    actions = reshape(rb.data.action, :, batch_size)
    logprobs = reshape(rb.data.logprob, :, batch_size)
    values = reshape(rb.data.value, :, batch_size)
    advantages = reshape(advantages, :, batch_size)
    returns = reshape(returns, :, batch_size)

    b_inds = 1:batch_size

    for epoch in 1:config.update_epochs
      b_inds = shuffle(b_inds)

      for start in 1:minibatch_size:batch_size
        pg_loss = 0.0
        v_loss = 0.0
        entropy_loss = 0.0

        loss, gs = Flux.withgradient((actor, critic)) do (actor_model, critic_model)
          end_ind = start + minibatch_size - 1
          mb_inds = b_inds[start:end_ind]

          mb_states = @view states[:, mb_inds]
          mb_actions = vec(@view actions[:, mb_inds])
          mb_advantages = @view advantages[mb_inds]
          mb_logprobs = vec(@view logprobs[mb_inds])
          mb_values = @view values[mb_inds]
          mb_returns = @view returns[mb_inds]

          newlogprob, entropy = logprob_actions(mb_states, actor_model, mb_actions)
          newvalue = critic_model(mb_states)
          newlogprob = vec(newlogprob)
          newvalue = vec(newvalue)

          # policy loss
          mb_advantages = if config.normalize_advantages
            # todo: revisit fused vector ops in julia perf tips
            (mb_advantages .- mean(mb_advantages)) ./ (std(mb_advantages) .+ 1e-8)
          else
            mb_advantages
          end

          raw_logratio = newlogprob - mb_logprobs
          n_clipped = count(x -> abs(x) > 20f0, raw_logratio)
          n_clipped > 0 && @warn "logratio clipped" n_clipped max_abs_logratio = maximum(abs.(raw_logratio))
          logratio = clamp.(raw_logratio, -20f0, 20f0)
          ratio = exp.(logratio)
          pg_loss1 = @. -mb_advantages * ratio
          pg_loss2 = @. -mb_advantages * clamp.(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
          pg_loss = mean(max.(pg_loss1, pg_loss2))

          # value loss
          v_loss = if config.clip_value_loss
            v_loss_unclipped = (newvalue .- mb_returns) .^ 2
            # todo: revisit fused vector ops in julia perf tips
            v_clipped = @. mb_values + clamp(newvalue - mb_values, -config.clip_coef, config.clip_coef)
            v_loss_clipped = @. (v_clipped - mb_returns)^2
            v_loss_max = max.(v_loss_unclipped, v_loss_clipped)
            0.5 * mean(v_loss_max)
          else
            0.5 * mean((newvalue - mb_returns) .^ 2)
          end

          entropy_loss = mean(entropy)
          pg_loss - config.ent_coeff * entropy_loss + config.v_coef * v_loss
        end

        log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
        @info "Training Statistics" loss pg_loss v_loss entropy_loss log_step_increment = log_step_inc
        last_log_step = deepcopy(global_step)

        Flux.update!(opt_state, (actor, critic), gs[1])
      end
    end
  end
  actor, critic
end

function ppo(config::PPOConfig=PPOConfig())
  throw(ArgumentError("ppo now requires a POMDPs.MDP or env factory, for example ppo(mdp, config) or ppo(() -> MDPEnv(mdp), config)."))
end
