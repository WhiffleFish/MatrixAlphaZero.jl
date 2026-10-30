# Transfer-weight ablation

Two complementary sweeps over the transfer weight `w` from Theorem 1.

| Part | Script | Question | Metric |
|---|---|---|---|
| A | `dubin/dubin-2026-07-19/tabular_transfer_weight_sweep.jl` | Does the best `w` shrink with transfer error, as Theorem 1 and Reviewer 5's threshold predict? | Exact finite-tree Nash-gap reduction versus restarting (Figure 2 setting) |
| B | `dubin/dubin-2026-07-20/benchmark_transfer_weight_sweep.jl`, `sda/sda-2026-07-21/benchmark_transfer_weight_sweep.jl` | How sensitive is the deployed SM-AZ to `w` and the search budget? | Seat-balanced utility of SM-AZ against SM-AZ(−RT) |

## How `w` maps to the code

`prior_scale = w * T1` (see `effective_prior_mass` in `src/search/mcts.jl`).

- **Part A**: each node is solved as its own root, so `w` is exactly the theorem's weight. `T1 = 64`, `prior_scale = w * 64`, and `w = 1` with the `full` variant is Figure 2. The sweep reproduces Figure 2's trial gaps exactly at `w = 1`.
- **Part B**: `T1 = 500`, the training search budget that produced the regret targets, so `w = prior_scale / 500`. The deployed `prior_scale = 2.5` is `w = 0.005`. Below the root the mass is further attenuated by learned reach `q(h)`, so `w` is the root-level weight.
- **Variants**: Part B transfers regret only, which is what the paper's tables use. Part A runs both `full` (regret, average strategy and pseudo-counts, as in Figure 2) and `regret_only`.

## Default grids

**Part A**
- `w`: 1/64, 1/16, 1/4, 1/2, 1, 2, 4. The restart baseline is `w = 0`.
- `δR = δσ`: 0, 0.05, 0.1, 0.2, 0.4.
- `δV = 0`, `T1 = T2 = 64`, 32 paired trials.

**Part B**
- `prior_scale`: 0, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, i.e. `w` from 0 to 1.
- Queries: 25, 50, 100, 200.
- Each grid point is played in both seats, giving 80 cells per domain.
- Games per seat: 1000 for Dubins, 2000 for SDA.
- `prior_scale = 0` is the same search as SM-AZ(−RT). It gives the noise floor and the per-seat reference.
- Evaluation uses a fresh seed (20261030):
  - **SDA**: initial states come from the core LEO distribution.
  - **Dubins**: initial states default to `burnin:8`. Each run starts from the Table 2 state and takes 0–8 uniform-random joint actions, so the states stay near the self-play distribution without using a single fixed state. Pass `--initial-states reference` to repeat the Table 2 protocol exactly.

## Setup on the server

```bash
git clone git@github.com:WhiffleFish/MatrixAlphaZero.jl.git
```

```bash
cd MatrixAlphaZero.jl && git checkout new-exploitability
```

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
```

The model files are not tracked in git. Copy them from this machine, run from the repository root here (`SERVER` is a placeholder for your host):

```bash
rsync -avR experiments/dubin/dubin-2026-07-20/oracle_rm_plus_no_transfer_train.jld2 experiments/dubin/dubin-2026-07-20/models_rm_plus_no_transfer_train/oracle1221.jld2 experiments/sda/sda-2026-07-21/oracle_rm_plus_no_transfer_train_mean_leo.jld2 experiments/sda/sda-2026-07-21/models_rm_plus_no_transfer_train_mean_leo/oracle1221.jld2 experiments/sda/sda-2026-07-21/regret_fit_results_softplus_long/models.jld2 SERVER:MatrixAlphaZero.jl/
```

Pass `--checkpoint 1221` in Part B so the run cannot pick a different "latest" checkpoint on the server.

## Smoke tests (also precompile and resolve CondaPkg)

Run these one at a time before launching shards. Many processes starting at once otherwise queue on the CondaPkg lock and the precompile cache.

```bash
julia -t 4 --project=experiments experiments/dubin/dubin-2026-07-19/tabular_transfer_weight_sweep.jl --test
```

```bash
julia --project=experiments experiments/dubin/dubin-2026-07-20/benchmark_transfer_weight_sweep.jl --test --checkpoint 1221
```

```bash
julia --project=experiments experiments/sda/sda-2026-07-21/benchmark_transfer_weight_sweep.jl --test --checkpoint 1221
```

Each run writes to a `smoke/` subfolder of its output directory. Delete those folders afterwards.

## Launch

Logs go under `logs/`. Run each launch inside `tmux` or with `nohup` so it survives disconnects.

```bash
mkdir -p logs/weight_sweep
```

### Part A: tabular (threads over tree nodes)

Main grid, about 4 s per trial on 4 threads:

```bash
nohup julia -t auto --project=experiments experiments/dubin/dubin-2026-07-19/tabular_transfer_weight_sweep.jl > logs/weight_sweep/tabular.log 2>&1 &
```

Optional budget slice for Reviewer 5's crossover condition (`T2` dependence):

```bash
nohup julia -t auto --project=experiments experiments/dubin/dubin-2026-07-19/tabular_transfer_weight_sweep.jl --target-iterations 16,64,256 --transfer-errors 0,0.1,0.2 --output experiments/dubin/dubin-2026-07-19/tabular_transfer_weight_results_budget > logs/weight_sweep/tabular_budget.log 2>&1 &
```

### Part B: learned priors (single-threaded rollouts, so shard across processes)

Search and rollouts are serial, so give each process one core. Use at most the number of physical cores. Each script sets BLAS to one thread when `--shard` has more than one part.

Dubins, main run with burn-in states (about 30 core-minutes in total):

```bash
N=8; for i in $(seq 1 $N); do nohup julia --project=experiments experiments/dubin/dubin-2026-07-20/benchmark_transfer_weight_sweep.jl --checkpoint 1221 --shard $i/$N > logs/weight_sweep/dubin_shard$i.log 2>&1 & done
```

Dubins with the exact Table 2 protocol (fixed state, fresh seed), for direct comparison:

```bash
N=8; for i in $(seq 1 $N); do nohup julia --project=experiments experiments/dubin/dubin-2026-07-20/benchmark_transfer_weight_sweep.jl --checkpoint 1221 --initial-states reference --output-dir experiments/dubin/dubin-2026-07-20/transfer_weight_sweep_reference --shard $i/$N > logs/weight_sweep/dubin_ref_shard$i.log 2>&1 & done
```

SDA (about 12–13 core-hours in total; the longest single cell, at 200 queries, takes about 20 min):

```bash
N=32; for i in $(seq 1 $N); do nohup julia --project=experiments experiments/sda/sda-2026-07-21/benchmark_transfer_weight_sweep.jl --checkpoint 1221 --shard $i/$N > logs/weight_sweep/sda_shard$i.log 2>&1 & done
```

Shards take every `N`-th cell after sorting the cells from most to fewest queries, so the work is roughly balanced. All shards of one sweep must use the same `N` and the same options.

## Monitoring

Part B cells finished (out of 80 per domain):

```bash
ls experiments/sda/sda-2026-07-21/transfer_weight_sweep/cells/*.csv | wc -l
```

```bash
ls experiments/dubin/dubin-2026-07-20/transfer_weight_sweep/cells/*.csv | wc -l
```

Latest line from every shard, including the per-shard ETA:

```bash
for f in logs/weight_sweep/sda_shard*.log; do echo "$f: $(grep -h 'cell \|ERROR' $f | tail -1)"; done
```

Live processes:

```bash
pgrep -fa transfer_weight_sweep | wc -l
```

Errors anywhere:

```bash
grep -l "ERROR" logs/weight_sweep/*.log
```

Part A progress (one line per finished trial, with ETA):

```bash
grep "trial" logs/weight_sweep/tabular.log | tail -3
```

Each shard rewrites `summary.csv` when it finishes. To refresh the summaries and draft figures from partial results at any time:

```bash
julia --project=experiments experiments/plot_transfer_weight_sweeps.jl
```

## Resuming

Rerun the same command with the same `--shard` split.
- **Part B**: a cell is written atomically when it finishes, so finished cells are skipped and interrupted cells start over.
- **Part A**: whole trials are appended to `trials.csv`, and finished trials are skipped.

Changing `--runs`, `--seed`, `--initial-states`, the grid, or the trial settings requires a new `--output-dir`/`--output`. The scripts do not check old cells against new settings.

## Outputs

| Path | Contents |
|---|---|
| `dubin/dubin-2026-07-19/tabular_transfer_weight_results/trials.csv` | One row per trial × variant × `δV` × `δR` × `T2` × `w` |
| `.../tabular_transfer_weight_results/summary.csv` | Means and SEMs, plus Reviewer 5's threshold `w ≤ T1Δu²/(Δu√T1 + T1δR)²` computed with the root payoff range (`config.txt` records the root, positive-median and max node ranges) |
| `<domain>/transfer_weight_sweep/cells/*.csv` | One file per Part B matchup, with the full settings and checkpoint |
| `<domain>/transfer_weight_sweep/summary.csv` | Seat-balanced utility, 95% CI, per-seat utility of the transfer solver, and per-seat delta against the `prior_scale = 0` cell |
| `transfer_weight_figures/` | `tabular_transfer_weight_{full,regret_only}.pdf` and `learned_transfer_weight.pdf` |

Copy the result folders back with `rsync` and rerun the plot script locally. It skips any input that is missing.

## Sanity checks once results are in

- **Part A**: the `full`, `w = 1`, `δR = 0` row should match Figure 2: transfer gap about 0.150 and improvement about 0.101.
- **Part B**: the `prior_scale = 0` rows should be within noise of zero.
- **Part B**: the 50-query, `prior_scale = 2.5` row is the Table 2/3 SM-AZ versus SM-AZ(−RT) matchup on new states and seeds. Expect it to be smaller than the selection-affected Dubins table value (+0.05).
