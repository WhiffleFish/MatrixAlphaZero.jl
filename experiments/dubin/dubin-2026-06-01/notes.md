# Dubin Training With Stat Rollout Evaluation

Train Dubin with the current fitted regret-transfer SM-OOS pipeline and log
aggregate `StatRolloutSimulator` matchup metrics to W&B during training.

```sh
julia --project=experiments experiments/dubin/dubin-2026-06-01/train.jl
```

Quick smoke run:

```sh
julia --project=experiments experiments/dubin/dubin-2026-06-01/train.jl --test
```

Logged stat rollout metrics are namespaced under:

- `stat_rollout/az_p1_vs_heuristic/*`
- `stat_rollout/heuristic_vs_az_p2/*`

Each namespace includes mean reward, reward standard error, mean steps, and
Dubin outcome rates for attacker goal, tag, and timeout.
