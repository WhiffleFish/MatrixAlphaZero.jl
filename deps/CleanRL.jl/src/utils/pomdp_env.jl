export MDPEnv, state, step!, reset!, is_terminated, current_reward,
  action_count, action_dim, state_dim, random_action, single_state_dim,
  single_action_count

mutable struct MDPEnv{M,RNG,S,A,SE,AD,RS}
  mdp::M
  rng::RNG
  state::S
  reward::Float32
  terminal::Bool
  step_count::Int
  max_steps::Union{Nothing,Int}
  actions::Vector{A}
  state_encoder::SE
  action_decoder::AD
  random_action_sampler::RS
end

_as_float_vector(x::AbstractArray) = Float32.(vec(x))
_as_float_vector(x::Number) = Float32[x]

function MDPEnv(
  mdp::POMDPs.MDP;
  rng=Random.default_rng(),
  max_steps::Union{Nothing,Int}=nothing,
  actions=collect(POMDPs.actions(mdp)),
  state_encoder=s -> POMDPs.convert_s(Vector{Float32}, s, mdp),
  action_decoder=a -> actions[Int(a)],
  random_action_sampler=rng -> rand(rng, actions),
)
  s = rand(rng, POMDPs.initialstate(mdp))
  terminal = POMDPs.isterminal(mdp, s)
  MDPEnv(mdp, rng, s, 0.0f0, terminal, 0, max_steps, actions,
    state_encoder, action_decoder, random_action_sampler)
end

state(env::MDPEnv) = env.state_encoder(env.state)
current_reward(env::MDPEnv) = env.reward
is_terminated(env::MDPEnv) = env.terminal
action_count(env::MDPEnv) = length(env.actions)
state_dim(env::MDPEnv) = length(state(env))
action_dim(env::MDPEnv) = length(_as_float_vector(random_action(env)))
random_action(env::MDPEnv) = env.random_action_sampler(env.rng)

function reset!(env::MDPEnv)
  env.state = rand(env.rng, POMDPs.initialstate(env.mdp))
  env.reward = 0.0f0
  env.terminal = POMDPs.isterminal(env.mdp, env.state)
  env.step_count = 0
  env
end

function step!(env::MDPEnv, action)
  env.terminal && return env
  decoded_action = env.action_decoder(action)
  sp, r = POMDPs.@gen(:sp, :r)(env.mdp, env.state, decoded_action, env.rng)
  env.state = sp
  env.reward = Float32(r)
  env.step_count += 1
  env.terminal = POMDPs.isterminal(env.mdp, sp) ||
    (env.max_steps !== nothing && env.step_count >= env.max_steps)
  env
end

function single_state_dim(env)
  state_dim(first(env.envs))
end

function single_action_count(env)
  action_count(first(env.envs))
end
