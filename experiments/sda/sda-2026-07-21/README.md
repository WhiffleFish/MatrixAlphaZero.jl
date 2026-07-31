# SDA experiment

This directory contains the Space Domain Awareness experiment reported in the
paper. It compares three depth-5 RM+ search policies in a seat-balanced round
robin:

- `zero_oracle`: zero leaf values and no transfer;
- `value_oracle`: the learned value function and no transfer;
- `full_solver`: learned values plus inference-only regret transfer.

All conditions use 50 search queries, mean backup, search-value targets,
search exploration `0.1`, action exploration `0`, and at most 50 environment
steps. The full solver uses regret-only transfer with prior scale `2.5`.
Initial states follow the correlated LEO distribution in `initial_state.jl`.

## Reproduce the round robin

From the repository root:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_solver_round_robin.jl
```

The default run evaluates every ordered matchup on 1,000 paired initial
states and writes results to `solver_round_robin_q50_scale2p5/`. For a quick
end-to-end check:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_solver_round_robin.jl \
  --test --output-dir /tmp/sda-round-robin-smoke
```

The benchmark requires the retained oracle architecture, iteration-1221
checkpoint, and fitted softplus regret model. `train.jl` defines the original
training run; `generate_regret_fit_dataset.jl` and `fit_regret_softplus.jl`
define the post-training regret refit.

The committed CSV files contain the paper's 1,000-rollout results. Their
`checkpoint` field uses a repository-relative artifact path.
