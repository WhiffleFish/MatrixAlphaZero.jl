Base.@kwdef struct DQNConfig
  run_name::String = format(now(), "yy-mm-dd|HH:MM:SS")
  log_dir::String = "logs"

  log_frequencey::Int = 1000

  total_timesteps::Int = 500_000

  buffer_size::Int64 = 10_000
  min_buff_size::Int64 = 200

  lr::Float64 = 0.0001
  train_freq::Int64 = 10
  target_net_freq::Int64 = 100
  batch_size::Int64 = 120
  gamma::Float64 = 0.99

  epsilon_start::Float64 = 1.0
  epsilon_end::Float64 = 0.05
  epsilon_duration::Float64 = 10_000
end

function make_nn(env)
  in_size = state_dim(env)
  out_size = action_count(env)
  Chain(Dense(in_size, 120, relu), Dense(120, 84, relu), Dense(84, out_size))
end

function linear_schedule(start_ϵ, end_ϵ, duration, t)
  slope = (end_ϵ - start_ϵ) / duration
  max(slope * t + start_ϵ, end_ϵ)
end


function dqn(mdp::POMDPs.MDP, config::DQNConfig=DQNConfig(); kwargs...)
  dqn(MDPEnv(mdp; kwargs...), config)
end

function dqn(env::MDPEnv, config::DQNConfig=DQNConfig())
  Logger.make_logger("dqn|$(config.run_name)"; log_dir=config.log_dir)

  q_net = make_nn(env)
  target_net = deepcopy(q_net)
  opt = Adam(config.lr)
  opt_state = Flux.setup(opt, q_net)

  transition = (
    state=rand(Float32, state_dim(env)),
    action=[rand(1:action_count(env))],
    reward=1.0,
    next_state=rand(Float32, state_dim(env)),
    terminal=true
  )
  rb = Buffer.ReplayBuffer(transition, config.buffer_size)

  ϵ_schedule = t -> linear_schedule(config.epsilon_start, config.epsilon_end, config.epsilon_duration, t)

  episode_return = 0
  episode_length = 0

  start_time = time()
  reset!(env)
  for global_step in 1:config.total_timesteps
    obs = deepcopy(state(env))  # state needs to be coppied otherwise state and next_state is the same
    # action selection
    ϵ = ϵ_schedule(global_step)
    action = if rand() < ϵ
      rand(1:action_count(env))
    else
      qs = q_net(obs)
      argmax(qs)
    end

    step!(env, action)

    # add to buffer
    transition = (
      state=obs,
      action=[action],
      reward=[current_reward(env)],
      next_state=deepcopy(state(env)),
      terminal=[is_terminated(env)]
    )
    Buffer.add!(rb, transition)

    # Recording episode statistics
    episode_return += current_reward(env)
    episode_length += 1
    if is_terminated(env)
      steps_per_sec = trunc(global_step / (time() - start_time))
      @info "Episode Statistics" episode_return episode_length global_step ϵ steps_per_sec
      episode_length, episode_return = 0, 0

      reset!(env)
    end

    # Learning
    if (global_step > config.min_buff_size) && (global_step % config.train_freq == 0)
      data = Buffer.sample(rb, config.batch_size)
      # Convert actions to CartesianIndexes so they can be used to index q matrix
      actions = CartesianIndex.(vec(data.action), 1:length(data.action))

      next_q = data.next_state |> target_net |> eachcol .|> maximum
      td_target = vec(data.reward) + config.gamma * next_q .* (1.0 .- vec(data.terminal))

      # Get grads and update model
      loss, gs = Flux.withgradient(q_net) do net
        q = net(data.state)
        q = q[actions]
        Flux.mse(td_target, q)
      end
      Flux.update!(opt_state, q_net, gs[1])

      if global_step % config.target_net_freq == 0
        target_net = deepcopy(q_net)
      end

      if global_step % config.log_frequencey == 0
        @info "Training Statistics" loss
      end
    end
  end
  q_net
end

function dqn(config::DQNConfig=DQNConfig())
  throw(ArgumentError("dqn now requires a POMDPs.MDP or MDPEnv, for example dqn(mdp, config)."))
end
