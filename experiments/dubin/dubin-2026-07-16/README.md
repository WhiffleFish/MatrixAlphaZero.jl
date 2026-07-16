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
