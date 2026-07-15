# Dubin Independent-Network RM+ Transfer Experiment

This experiment tests whether fitted regret and strategy transfer improves a
500-query RM+ search on Dubin relative to an otherwise matched value-only RM+
baseline when no learned state representation is shared between objectives.

The critic, each player's regret predictor, and each player's strategy
predictor are separate four-layer MLPs that consume the raw eight-dimensional
state. There is no learned trunk, so value, regret, and strategy gradients
cannot compete through shared parameters.

## Conditions

| Launcher | Oracle | Regret scale | Strategy scale |
|---|---|---:|---:|
| `train_value_only.jl` | `CriticOnly` | 0 | 0 |
| `train_transfer_default.jl` | `FittedRegretModel` | 0.25 | 1.0 |
| `train_transfer_conservative.jl` | `FittedRegretModel` | 0.0625 | 0.25 |

All conditions use RM+, 500 tree queries, depth 5, the same seed, identical
initial critic weights, the same optimizer schedule, and the same heuristic
evaluation protocol. Every end-to-end network has shape
`8 -> 32 -> 32 -> 32 -> output`, matching the depth of the previous composed
trunk-plus-head paths. The conservative arm tests the lower transfer mass that
remained beneficial with noisy fits in the sequential toy benchmark.

## Launch

Run each condition from the repository root:

```bash
julia --project=experiments experiments/dubin/dubin-2026-07-15/train_value_only.jl
julia --project=experiments experiments/dubin/dubin-2026-07-15/train_transfer_default.jl
julia --project=experiments experiments/dubin/dubin-2026-07-15/train_transfer_conservative.jl
```

For a local smoke test, append `--test`.

All W&B runs use group
`dubin-2026-07-15-independent-rm-plus-transfer-500`, with the condition and
independent-network architecture encoded in the run name and config.

## Comparison

Use the two controlled-player heuristic rewards as primary outcomes:

- `eval/az_p1_vs_heuristic/reward`
- `eval/heuristic_vs_az_p2/reward`

Compare matched iterations and use their logged standard errors. For transfer
runs, inspect these mechanism metrics alongside reward:

- `progress/transfer_regret_confidence`
- `progress/transfer_strategy_confidence`
- `progress/transfer_regret_mass`
- `progress/transfer_strategy_mass`

The transfer claim is supported only if a transfer arm improves the heuristic
evaluation without relying on a transient early peak and without collapsing one
player side.
