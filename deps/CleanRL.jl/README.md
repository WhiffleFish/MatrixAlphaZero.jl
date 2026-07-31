# CleanRL.jl
Simple single file implementations of Reinforcement Learning algorithms in Julia.  
Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl)

## POMDPs.jl MDPs
Algorithms now take a `POMDPs.MDP` or `MDPEnv` instead of constructing hardcoded
`ReinforcementLearning.jl` environments internally:

```julia
using CleanRL
using POMDPs

dqn(mdp, DQNConfig(total_timesteps=100_000))
ppo(mdp, PPOConfig(num_envs=8))
```

`MDPEnv` uses `POMDPs.convert_s(Vector{Float32}, s, mdp)` to turn model states
into neural-network observations. Define that method for struct states:

```julia
POMDPs.convert_s(::Type{Vector{Float32}}, s::MyState, ::MyMDP) = Float32[
  s.x,
  s.y,
  s.v,
]
```

Discrete algorithms enumerate `actions(mdp)` and internally use integer action
indices. For custom or continuous action spaces, construct `MDPEnv` directly
with `actions`, `action_decoder`, and `random_action_sampler`.

## TODO (Algorithms):
* ~~Simple DQN~~
* ~~A2C~~
* Rainbow
* ~~PPO~~
* ~~DDPG~~
* SAC

## TODO (Utils):
* ~~General replay buffer~~
* ~~CLI for hyperparameters~~
* ~~Support loggers~~
* GPU training
* ~~Multi-thread PPO~~
* Vectorized envs
* ~~Plotting~~
* ~~Multi-loggers (file/console/Tensorboard)~~
* ~~Make nn inputs F32 - F32 env wrapper?~~ - done for PPO
* Make individual file runners e.g experiments/run_ppo.(jl/sh)
* Better logging - log interval and always log at correct step

## TODO (Investigate):
* Profile PPO
