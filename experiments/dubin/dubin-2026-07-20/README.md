# Dubin training and solver cross-play

This directory contains the Dubin training configuration and the direct solver
cross-play experiment reported in the AAAI-2027 paper. The comparison uses one
trained value oracle for all three deployment solvers:

- `zero_oracle`: RM+ tree search with a zero value oracle and no transfer;
- `value_oracle`: RM+ tree search with the learned value oracle and no transfer;
- `full_solver`: the same value oracle with regret-only initialization.

All paper evaluations use 50 tree queries, depth 5, mean backup, search epsilon
0.1, action epsilon 0.0, regret prior scale 2.5, and 1,000 rollouts per ordered
matchup. The final model is checkpoint 1221.

## Required model files

The included evaluation requires:

```text
oracle_rm_plus_no_transfer_train.jld2
models_rm_plus_no_transfer_train/oracle1221.jld2
```

The first file stores the model architecture and the second stores the final
parameters.

## Reproduce the paper table

Run from the repository root:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_solver_round_robin.jl
```

Reference results are stored in `solver_round_robin_q50_scale2p5/` as the
detailed ordered matchups and the player-1 utility and standard-error matrices.

For a fast compatibility check:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_solver_round_robin.jl \
  --test --output-dir /tmp/dubin-round-robin-smoke
```

## Train from scratch

Run the full training configuration with:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/train.jl
```

Use `--test` for a short training smoke test. Training uses fresh,
zero-initialized RM+ searches; regret transfer is enabled only during the
deployment evaluation.
