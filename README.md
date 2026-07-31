# MatrixAlphaZero supplementary material

This repository contains the code and saved model artifacts needed to
reproduce the experiments reported in the paper. The retained experimental
results are:

1. a network-free Dubin Tag regret-transfer study;
2. a Dubin Tag round robin comparing zero-oracle, value-oracle, and
   regret-transfer search; and
3. the corresponding solver round robin in the Space Domain Awareness (SDA)
   domain.

All commands below should be run from the repository root.

## Setup

Install Julia 1.11 or newer, then instantiate the experiment environment:

```sh
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
```

The packages developed specifically for these experiments are included under
`deps/` and selected through repository-relative entries in
`experiments/Project.toml`. No separate checkout of those packages is needed.

The experiments run on CPU and do not require a
GPU. Full evaluations are substantially slower than the smoke tests because
they perform many independent tree searches and rollouts.

## Quick end-to-end checks

These commands use reduced search budgets and rollout counts. They verify that
the environment, saved models, planners, and output writers work together;
they do not reproduce the numerical paper results.

```sh
# Network-free Dubin Tag experiment
julia --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl \
  --test --output /tmp/dubin-transfer-smoke

# Dubin Tag solver round robin
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_solver_round_robin.jl \
  --test --output-dir /tmp/dubin-round-robin-smoke

# SDA solver round robin
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_solver_round_robin.jl \
  --test --output-dir /tmp/sda-round-robin-smoke
```

## Reproduce the paper experiments

### 1. Network-free Dubin Tag regret transfer

Run:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl
```

Outputs are written to:

```text
experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_results/
```

The main numerical results are in `summary.csv` and `trials.csv`; the plotted
paper result is `regret_transfer_heatmaps.pdf`. The checked-in files in this
directory are the reference outputs used for the paper.

### 2. Dubin Tag solver round robin

Run:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_solver_round_robin.jl
```

The experiment evaluates every ordered pairing of:

- `zero_oracle`: RM+ search with zero leaf values and no transfer;
- `value_oracle`: RM+ search with the learned value function and no transfer;
- `full_solver`: the same learned value function with regret-only transfer.

All conditions use 50 tree queries, depth 5, mean backup, search epsilon 0.1,
action epsilon 0, and 1,000 paired rollouts. The full solver uses prior scale
2.5. The retained architecture and iteration-1221 checkpoint are:

```text
experiments/dubin/dubin-2026-07-20/oracle_rm_plus_no_transfer_train.jld2
experiments/dubin/dubin-2026-07-20/models_rm_plus_no_transfer_train/oracle1221.jld2
```

Outputs and reference results are stored in:

```text
experiments/dubin/dubin-2026-07-20/solver_round_robin_q50_scale2p5/
```

`matchups.csv` contains the detailed ordered matchups. `p1_utilities.csv` and
`p1_stderrs.csv` contain the matrices used to construct the paper table.

### 3. SDA solver round robin

Run:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_solver_round_robin.jl
```

This experiment compares the same three solver conditions in the SDA domain.
It uses 50 tree queries, depth 5, mean backup, search epsilon 0.1, action
epsilon 0, prior scale 2.5, a 50-step episode limit, and 1,000 paired rollouts
per ordered matchup. Initial states are sampled from the correlated low-Earth
orbit distribution defined in `initial_state.jl`.

The evaluation uses the retained architecture, iteration-1221 checkpoint, and
post-training softplus regret fit:

```text
experiments/sda/sda-2026-07-21/oracle_rm_plus_no_transfer_train_mean_leo.jld2
experiments/sda/sda-2026-07-21/models_rm_plus_no_transfer_train_mean_leo/oracle1221.jld2
experiments/sda/sda-2026-07-21/regret_fit_results_softplus_long/models.jld2
```

Outputs and reference results are stored in:

```text
experiments/sda/sda-2026-07-21/solver_round_robin_q50_scale2p5/
```

In addition to the detailed matchup and player-1 matrices, this directory
contains the seat-balanced utility and standard-error matrices reported in the
paper.

## Training and regret refitting

The saved checkpoints above are sufficient to rerun the reported evaluations;
training is not required. For completeness, the original training entrypoints
are:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/train.jl

julia --project=experiments \
  experiments/sda/sda-2026-07-21/train.jl
```

The SDA post-training regret model is produced in two stages:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/generate_regret_fit_dataset.jl

julia --project=experiments \
  experiments/sda/sda-2026-07-21/fit_regret_softplus.jl
```

Training and dataset generation are considerably more expensive than loading
the supplied checkpoints.

## Reproducibility notes

- Each script defines a fixed default seed and accepts command-line overrides
  for its principal evaluation settings.
- Round-robin conditions share their sampled initial-state bank, making the
  comparisons paired across solver conditions.
- Reference CSV files are included so reproduced results can be compared
  directly with the values used in the paper.
- Additional experiment-specific details are documented in the README files
  beside each experiment.
