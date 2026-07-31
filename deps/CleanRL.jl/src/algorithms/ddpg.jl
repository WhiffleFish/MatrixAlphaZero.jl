Base.@kwdef struct DDPGConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")
  log_dir::String = "logs"

  total_timesteps::Int = 5_000_000
  buffer_size::Int64 = 1e5
  min_buff_size::Int64 = 200
  batch_size::Int64 = 120
  warmup_steps::Int64 = 2500
  log_frequencey::Int = 1000

  lr::Float64 = 0.0001
  tau::Float64 = 0.005
  gamma::Float64 = 0.99
  exploration_noise::Float64 = 0.1
end

# todo: how to work with ClosedInterval to get correct action shape
# todo: move this to the networks class and create a NormalActorHead
function make_ddpg_nn(env, exploration_noise::Float64)
  ob_size = state_dim(env)
  ac_size = action_dim(env)

  actor = Chain(
    Dense(ob_size, 64, tanh_fast),
    Dense(64, 64, tanh_fast),
    Dense(64, ac_size, tanh_fast),
    # Does .+ broadcast the same noise over all batches?
    x -> x .* 2 .+ (randn(ac_size) * exploration_noise)
  )
  critic = Chain(
    Dense(ob_size + ac_size, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
  )

  actor, critic
end

function polyak_update!(target::Zygote.Params, source::Zygote.Params, tau::Float64)
  for (t, s) in zip(target, source)
    t .= tau .* s .+ (1.0 - tau) .* t
  end
end

get_q(critic, obs, act) = critic(vcat(obs, act))

function ddpg(mdp::POMDPs.MDP, config::DDPGConfig=DDPGConfig(); kwargs...)
  ddpg(MDPEnv(mdp; kwargs...), config)
end

function ddpg(env::MDPEnv, config::DDPGConfig=DDPGConfig())
  Logger.make_logger("ddpg|$(config.run_name)"; log_dir=config.log_dir)

  actor, critic = make_ddpg_nn(env, config.exploration_noise)
  target_actor = deepcopy(actor)
  target_critic = deepcopy(critic)
  actor_opt = Adam(config.lr)
  critic_opt = Adam(config.lr)
  actor_opt_state = Flux.setup(actor_opt, actor)
  critic_opt_state = Flux.setup(critic_opt, critic)

  transition = (
    state=rand(Float32, state_dim(env)),
    action=_as_float_vector(random_action(env)),
    reward=1.0,
    next_state=rand(Float32, state_dim(env)),
    terminal=true
  )

  rb = Buffer.ReplayBuffer(transition, config.buffer_size)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)

  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same
    # action selection
    action = global_step > config.warmup_steps ? actor(obs) : _as_float_vector(random_action(env))
    step!(env, action)

    # add to buffer
    transition = (
      state=obs,
      action=action,
      reward=[current_reward(env)],
      next_state=deepcopy(state(env)),
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += transition.reward[1]
    episode_length += 1
    if transition.terminal[1]
      steps_per_sec = trunc(global_step / (time() - start_time))
      @info "Episode Statistics" episode_return episode_length steps_per_sec global_step
      episode_length, episode_return = 0, 0

      reset!(env)
    end

    # Learning
    # todo too many transposes?
    if (global_step > config.min_buff_size) && (global_step > config.warmup_steps)
      data = Buffer.sample(rb, config.batch_size)

      next_acts = target_actor(data.next_state)
      next_q = get_q(target_critic, data.next_state, next_acts)
      td_target = data.reward + config.gamma * next_q .* (1.0 .- data.terminal)

      # critic update
      critic_loss, gs = Flux.withgradient(critic) do model
        q = get_q(model, data.state, data.action)
        Flux.mse(td_target, q)
      end
      Flux.update!(critic_opt_state, critic, gs[1])

      # actor update
      actor_loss, gs = Flux.withgradient(actor) do model
        -mean(get_q(critic, data.state, model(data.state)))
      end
      Flux.update!(actor_opt_state, actor, gs[1])

      # Move towards target nets
      polyak_update!(Flux.params(target_actor), Flux.params(actor), config.tau)
      polyak_update!(Flux.params(target_critic), Flux.params(critic), config.tau)

      if global_step % config.log_frequencey == 0
        @info "Training Statistics" actor_loss critic_loss
      end
    end
  end
  actor, critic
end

function ddpg(config::DDPGConfig=DDPGConfig())
  throw(ArgumentError("ddpg now requires a POMDPs.MDP or MDPEnv. For continuous actions, pass MDPEnv(mdp; actions=..., action_decoder=identity, random_action_sampler=...)."))
end
