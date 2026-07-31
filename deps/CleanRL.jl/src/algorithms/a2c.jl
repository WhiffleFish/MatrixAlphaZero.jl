Base.@kwdef struct A2CConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")
  log_dir::String = "logs"

  lr::Float64 = 0.0001

  total_timesteps::Int = 1_000_000
  min_replay_size::Int = 512

  gamma::Float64 = 0.99
end


function discounted_future_rewards(rewards::Vector{T}, terminals::Vector{Bool}, final_value::T, γ::T) where {T<:AbstractFloat}
  future_rewards = zeros(eltype(rewards), size(rewards))
  future_rewards[1] = last(terminals) ? 0.0 : last(rewards) + γ * final_value

  reversed_rewards = reverse(@view rewards[1:end-1])  # would be nice if the reverse was also a view
  reversed_terminals = reverse(@view terminals[1:end-1])
  for (i, (r, t)) in enumerate(zip(reversed_rewards, reversed_terminals))
    future_rewards[i+1] += t ? 0.0 : r + γ * future_rewards[i]
  end

  reverse(future_rewards)
end

# TODO:
#  test harder envs
#  mulitple V train steps
#  normalise advantage
function a2c(mdp::POMDPs.MDP, config::A2CConfig=A2CConfig(); kwargs...)
  a2c(MDPEnv(mdp; kwargs...), config)
end

function a2c(env::MDPEnv, config::A2CConfig=A2CConfig())
  Logger.make_logger("a2c|$(config.run_name)"; log_dir=config.log_dir)

  actor, critic = Networks.make_actor_critic(action_count(env), state_dim(env))
  opt = Flux.OptimiserChain(ClipNorm(0.5), Adam(config.lr))
  actor_opt_state = Flux.setup(opt, actor)
  critic_opt_state = Flux.setup(opt, critic)

  transition = (
    state=rand(Float32, state_dim(env)),
    action=[rand(1:action_count(env))],
    reward=1.0,
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.min_replay_size * 2)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    # state needs to be coppied otherwise each loop will overwrite ref in replay buffer
    obs = deepcopy(state(env))
    probs = softmax(actor(obs))
    ac_dist = Categorical(probs)
    action = rand(ac_dist)

    step!(env, action)

    transition = (
      state=obs,
      action=[action],
      reward=[current_reward(env)],
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += transition.reward[1]
    episode_length += 1

    if is_terminated(env)  # todo: might be missing a final transition
      if rb.size > config.min_replay_size  # training
        data = map(x -> x[:, 1:rb.size], rb.data) # todo -1 or not here?
        final_value = critic(state(env))[1]
        discounted_rewards = discounted_future_rewards(vec(data.reward), vec(data.terminal), final_value, config.gamma)
        advantage = discounted_rewards - vec(critic(data.state))

        # critic update
        critic_loss, critic_gs = Flux.withgradient(critic) do model
          values = model(data.state)
          mean((discounted_rewards - vec(values)) .^ 2)
        end
        Flux.update!(critic_opt_state, critic, critic_gs[1])

        # actor update
        actor_loss, actor_gs = Flux.withgradient(actor) do model
          ac_probs = data.state |> model |> softmax
          ac_dists = Categorical.(eachcol(ac_probs), check_args=false)
          # todo: grad of logpdf is slow, manually do logsoftmax.
          log_probs = logpdf.(ac_dists, vec(data.action))
          -mean(log_probs .* advantage)
        end
        Flux.update!(actor_opt_state, actor, actor_gs[1])

        # logging
        @info "Training Statistics" actor_loss critic_loss

        Buffer.clear!(rb)
      end

      steps_per_sec = trunc(global_step / (time() - start_time))
      @info "Episode Statistics" episode_return episode_length global_step steps_per_sec
      # reset counters
      episode_length, episode_return = 0, 0
      reset!(env)
    end

  end
  actor, critic
end

function a2c(config::A2CConfig=A2CConfig())
  throw(ArgumentError("a2c now requires a POMDPs.MDP or MDPEnv, for example a2c(mdp, config)."))
end
