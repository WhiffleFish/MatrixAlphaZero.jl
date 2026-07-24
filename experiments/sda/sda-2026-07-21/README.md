# SDA 2026-07-21: correlated LEO regime

This experiment narrows the SDA task to one declared in-distribution orbital
regime instead of averaging over the full July 20 altitude/phase mixture.

## Initial-state distribution

- target altitude: 600--1200 km
- observer altitude relative to target: -300--300 km
- absolute observer/target phase separation: 10--120 degrees
- separation sign: uniform
- target phase: uniform around the orbit
- circular, coplanar orbits

The training distribution, fixed periodic-evaluation bank, and primary final
benchmarks must all use `core_initialstate_distribution(game)` from
`initial_state.jl`. The old broad distribution can be reported separately as
an out-of-distribution stress test, but it is not part of the primary table.

## Changes from `sda-2026-07-20`

- network width increases from 32 to 64 for each independent value, regret,
  and average-strategy network;
- regret-matching backups change from `:sample` to `:mean`;
- training still uses no regret, strategy, count, or value prior mass;
- deployed tree search uses 100 queries, search epsilon 0.1, action epsilon
  0.0, and compares prior scale 100 against prior scale 0;
- primary evaluation is restricted to the correlated LEO distribution above.

Width 64 is a measured capacity increase, not a claim that capacity was the
only failure mode. It is small compared with the cost of the 500-query search,
while giving each dense network roughly four times the hidden-layer parameter
count. The no-transfer comparison remains necessary to isolate whether learned
priors actually help.

## Launch

From the repository root:

```sh
julia --project=experiments experiments/sda/sda-2026-07-21/train.jl --addprocs 16
```

## Frozen regret-regression experiment

The regret-regression test is deliberately split into data generation and
fitting. Data generation runs once from the latest synced checkpoint, using a
frozen 500-query mean-backup RM+ solver, depth 5, 50-step trajectories, and the
correlated LEO initial-state distribution. Tree-search epsilon remains 0.1,
while environment action epsilon is increased to 0.3 so the trajectory roots
cover more of the final solver's reachable state distribution. Episode seeds
are fixed, and the saved train/validation/test indices split complete
trajectories rather than individual timesteps.

As in the actual no-transfer training path, the checkpoint critic supplies
leaf values and actions come from the newly solved local average strategy. The
learned strategy head is loaded and its predictions are saved in the dataset
for analysis, but it does not initialize or bias the data-collection search.
Only environment decision states are saved: every example is the root of its
own 500-query solve. Internal non-root tree states are deliberately excluded.

Generate the frozen 65,536-step final-iterate dataset:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/generate_regret_fit_dataset.jl \
  --workers 8
```

The generator displays a terminal progress bar and emits one durable
`[regret-data]` line after every eight-episode worker chunk with sample count,
percentage, elapsed time, throughput, ETA, and last-chunk duration. The output
is `regret_fit_dataset_final_iter.jld2`. `--steps`, `--search-epsilon`,
`--action-epsilon`, and `--chunk-episodes` remain available for controlled
variants.

Fit and compare the checkpoint regret head, a fresh copy of the existing
single-head regressor, and the gated log-magnitude hurdle model:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/fit_regret_hurdle.jl \
  --tau 0.001
```

Both fitted candidates start from fresh random initialization; checkpoint
predictions are retained only as an evaluation baseline. Training emits a
`[regret-fit]` line every epoch by default, including validation RMSE, early
stopping status, elapsed time, and a maximum-runtime ETA. Use `--log-every N`
to reduce logging frequency.

The hurdle model uses per-action BCE-with-logits gates, positive-only Huber
loss on `log1p(regret / tau)`, and a small Huber reconstruction loss on the
final gate-times-magnitude prediction. It uses the gate probability directly
at inference rather than hard-thresholding it. Results are written under
`regret_fit_results_final_iter/`; the immutable source batch is
`regret_fit_dataset_final_iter.jld2`.

## Final-iterate regret-transfer benchmark

`benchmark_refit_regret_transfer.jl` loads the final checkpoint twice and
replaces only the second copy's two regret heads with the fresh single-head
fits. It compares value-only search, the online checkpoint transfer prior, and
the refitted regret transfer prior against the no-burn heuristic in both player
positions. All arms share the same fixed initial-state bank, value network,
average-strategy network, 100-query mean-backup search, search epsilon 0.1,
zero action epsilon, depth 5, and 50-step rollout horizon.

Run the default 200-rollout benchmark at prior scale 100:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_refit_regret_transfer.jl
```

Use `--prior-scale 25` or another nonnegative value no larger than the query
budget for transfer-mass sensitivity. Results are written to
`regret_transfer_benchmark_final_iter/heuristic_matchups.csv`; pass
`--output-dir` to preserve additional scale settings.

## Regret-only deployment screen

`benchmark_regret_transfer_schemes.jl` decouples the inference prior into
regret, average-strategy, and count/value-statistic components. It also supports
raw, hard- or soft-thresholded, top-one, normalized, sharpened, and
gap-modulated regret predictions. The default candidate set is a screening
grid; use `--only` to run a comma-separated subset.

The matched screen and two validation banks found that the most robust
configuration is deliberately simple:

- use the 1000-epoch softplus single-head regret fits from
  `regret_fit_results_softplus_long/models.jld2`;
- transfer raw fitted regret with `prior_scale=5`;
- retain the original joint-policy reach attenuation
  (`prior_reach_power=1`);
- set average-strategy and count/value-statistic prior weights to zero;
- retain 100 queries, depth 5, mean backup, search epsilon 0.1, action epsilon
  0.0, and the checkpoint value oracle.

Across the independent 200- and 300-state banks (500 rollouts per player role),
the pooled mean rewards were:

| search player | value only | raw regret, scale 5 | delta |
|---|---:|---:|---:|
| player 1 | 18.441 | 21.831 | +3.389 |
| player 2 | -11.336 | -10.683 | +0.653 |

The player-1 improvement is larger than its conservative independent-arm
standard error (delta 3.389, SE 1.437). The player-2 result is positive on both
independent banks but remains statistically modest (pooled delta 0.653,
conservative SE 0.564). This supports calling the scheme competitive, not
claiming a uniform or certified improvement.

The old coupled scale-10 warm start was worse than value-only on the 200-state
validation bank in both roles (-0.389 and -0.651). Thus the useful signal is
specifically the low-mass regret initialization; learned strategy mass and
count/value pseudo-observations should remain disabled.

Reproduce the independent confirmation:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_regret_transfer_schemes.jl \
  --runs 300 \
  --seed 20260723 \
  --only value_only,baseline_raw_s5.0,baseline_soft0.1_s10 \
  --output-dir \
  experiments/sda/sda-2026-07-21/regret_transfer_scheme_confirmation300
```

The full validation CSVs are:

- `regret_transfer_scheme_validation200/heuristic_matchups.csv`
- `regret_transfer_scheme_confirmation300/heuristic_matchups.csv`

## Value-only versus regret-only head-to-head

`benchmark_value_vs_regret_head_to_head.jl` pits value-only search directly
against the selected raw-regret scale-5 configuration in both seat assignments.
Both solvers use the same checkpoint value oracle and otherwise matched
100-query search settings.

On 500 shared initial states per orientation:

| regret-only seat | value-only seat | regret-only reward |
|---|---|---:|
| player 1 | player 2 | 9.987 +/- 0.570 |
| player 2 | player 1 | -7.374 +/- 0.526 |

Because SDA is role-asymmetric, the appropriate cross-play summary averages the
regret-only reward over both seats. That seat-balanced advantage is
`1.306 +/- 0.388`; treating the two orientation estimates as independent gives
an approximate 95% interval of `[0.546, 2.066]`.

Reproduce with:

```sh
julia --project=experiments \
  experiments/sda/sda-2026-07-21/benchmark_value_vs_regret_head_to_head.jl \
  --runs 500
```

Results are written to `value_vs_regret_head_to_head/matchups.csv`, with the
seat-balanced calculation in `summary.csv`.
