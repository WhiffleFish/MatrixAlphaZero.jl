# Dubin RM+ Regret-Transfer Experiment

This is the compute-focused follow-up to the independent-network transfer run
from 2026-07-15. It keeps the same separate raw-state value, regret, and
strategy networks and the default loss-scaled RM+ regret/strategy transfer, but
trains the critic on the configured 500-query root search value.

The fitted-regret MCTS self-play path now treats value supervision explicitly:

- `value_target=:search` uses the root search value and is selected here.
- `value_target=:gae` retains lambda-GAE supervision from real trajectories for
  experiments that intentionally request it.

## Launch

From the repository root:

```bash
julia --project=experiments experiments/dubin/dubin-2026-07-16/train_transfer.jl
```

Append `--test` for a local smoke run.

W&B uses group `dubin-2026-07-16-rm-plus-transfer-500`. No explicit run name is
set, so W&B assigns its default generated name. The configuration records the
condition as `transfer_default`, the search as `rm_plus_transfer`, the
independent-network architecture, and `search/value_target=search`.

## PPO solver exploitability

`ppo_solver_exploitability.jl` trains PPO approximate best responses against
three otherwise-identical RM+ planners:

- `zero_oracle`: zero leaf values, uniform fallback, and no transfer;
- `value_oracle`: the checkpoint critic, uniform fallback, and no transfer;
- `full_solver`: the checkpoint critic plus learned regret and average-strategy
  transfer.

The value-only condition deliberately hides the learned regret and strategy
heads, so the comparison isolates the value oracle. The full solver defaults to
the final W&B confidences for this run. Since checkpoint files contain weights
but not transfer state, `--source-mass auto` reconstructs source mass as
`checkpoint_iteration * --train-tree-queries` (1221 × 500 for the final model).

Smoke-test all three solvers and both best-response players:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-16/ppo_solver_exploitability.jl \
  --test --output-dir /tmp/dubin-ppo-solver-smoke
```

A full evaluation:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-16/ppo_solver_exploitability.jl \
  --total-timesteps 500000 \
  --eval-runs 500 \
  --output-dir experiments/dubin/dubin-2026-07-16/ppo_solver_exploitability_results
```

The script writes:

- `best_response_utilities.csv`: every solver/player result;
- `exploitability_summary.csv`: the resulting NashConv and half-NashConv lower
  bounds;
- `failures.csv`: any failed PPO runs, while preserving successful results.

PPO produces approximate rather than certified best responses. A negative
NashConv estimate means at least one learned response underfit and should not be
interpreted as evidence of negative exploitability. For more reproducible
comparisons use `--num-envs 1`; with multiple threaded environments, search
sampling does not use PPO's environment RNG. The default `--initial-state
reference` measures exploitability from the fixed training state; use
`--initial-state game` for the game's initial-state distribution.
