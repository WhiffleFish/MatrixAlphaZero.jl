export MultiThreadEnv

using Base.Threads: @spawn

"""
    MultiThreadEnv(f, n::Int)

Wrap `n` independent `MDPEnv`s created by `f` and step them in parallel.
"""
struct MultiThreadEnv{E,S,R}
  envs::Vector{E}
  states::S
  rewards::R
  terminals::BitVector
end

function MultiThreadEnv(f, n::Int)
  MultiThreadEnv([f() for _ in 1:n])
end

function MultiThreadEnv(envs::Vector{<:MDPEnv})
  s_batch = reduce(hcat, state.(envs))
  r_batch = current_reward.(envs)
  t_batch = BitVector(is_terminated.(envs))
  MultiThreadEnv(envs, s_batch, r_batch, t_batch)
end

Base.getindex(env::MultiThreadEnv, i) = env.envs[i]
Base.length(env::MultiThreadEnv) = length(env.envs)
Base.iterate(env::MultiThreadEnv, state...) = iterate(env.envs, state...)

function Base.show(io::IO, ::MIME"text/markdown", env::MultiThreadEnv)
  print(io, "MultiThreadEnv($(length(env)) x $(nameof(typeof(env[1].mdp))))")
end

function step!(env::MultiThreadEnv, actions)
  N = ndims(actions)
  @sync for i in 1:length(env)
    @spawn begin
      action = N == 1 ? actions[i] : selectdim(actions, N, i)
      step!(env[i], action)
    end
  end
  env
end

function (env::MultiThreadEnv)(actions)
  step!(env, actions)
end

function reset!(env::MultiThreadEnv; is_force=false)
  @sync for i in 1:length(env)
    if is_force || is_terminated(env[i])
      @spawn reset!(env[i])
    end
  end
  env
end

function state(env::MultiThreadEnv)
  N = ndims(env.states)
  @sync for i in 1:length(env)
    @spawn selectdim(env.states, N, i) .= state(env[i])
  end
  env.states
end

function current_reward(env::MultiThreadEnv)
  env.rewards .= current_reward.(env.envs)
  env.rewards
end

function is_terminated(env::MultiThreadEnv)
  env.terminals .= is_terminated.(env.envs)
  env.terminals
end
