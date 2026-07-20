# SDA RM+ Regret-Transfer Experiment

This is the 2026-07-18 SDA follow-up to the Dubin fitted-regret runs. It keeps
the July-10 `SNRGameSimple` task and training budget, but changes the learner and
evaluation so that the experiment directly tests the lessons from Dubin.

## Changes from the Dubin run

- The value, regret, and average-strategy heads are independent 16-input MLPs;
  there is no shared trunk.
- Search uses RM+ and loss-scaled regret/strategy transfer.
- Strategy transfer is reduced from `1.0` to `0.25` pseudo-mass scale. The
  Dubin search often preserved the learned strategy too strongly; this leaves
  more of the 500-query result determined by fresh local search.
- Regret transfer retains the `0.25` scale because the frozen-tree Dubin
  diagnostic showed a useful local improvement from learned priors.
- The Dubin payoff projection bound of `2` is not reused. SDA rewards and
  continuation values have a different, much larger scale, so projection is
  disabled until an SDA bound is calibrated from observed root Q matrices.
- Training exploration still decays from `0.3` to `0.1`, but tree search now
  accumulates the unperturbed RM+ policy and applies epsilon only to traversal.
  Evaluation always uses `epsilon=0`.
- Evaluation uses 100 fixed random initial states with deterministic seeds and
  reports full transfer search, a matched value-only search, and the raw learned
  average strategy against the no-burn heuristic. This separates transfer,
  value-oracle, and policy-head quality.
- Training uses plain Adam, matching the July-10 SDA script. Gradient norms are
  logged before the update and were not clipped in `eager-shape-48`.
- The W&B config records the external SDAGames tree SHA and fixed-bank SNR
  feature quantiles, because the 16th feature is uncapped even though the reward
  is capped.

The critic remains supervised with `value_target=:search`. Dubin showed that
this fixes the target inconsistency and substantially improves value fit, but
does not by itself imply a better policy. The searched-policy and raw-policy
rollouts are therefore the decision metrics, not value loss alone.

## Launch

From the repository root:

```bash
julia --project=experiments experiments/sda/sda-2026-07-18/train_transfer.jl
```

Append `--test` for a local smoke run. W&B uses group
`sda-2026-07-18-rm-plus-transfer-500`.

## Evaluation terminology

Do not describe PPO response-policy evaluations as exact exploitability. The
appropriate outputs are per-player **PPO best-response utilities** and, when
the two utilities are summed, an **empirical NashConv lower-bound estimate**.
A negative sum indicates response-policy underfitting and is not meaningful
evidence about the solver's game-theoretic quality.

The no-burn heuristic is intentionally retained for continuity, but it is a
narrow opponent. Before making a competitive-policy claim, add at least one
stronger scripted opponent.

## PPO response utilities

`ppo_solver_response_utilities.jl` trains one PPO response policy for each
player against the zero-oracle, value-only, and full-transfer solvers. It uses
the final run confidences by default, SDA's `gamma=0.98`, deployed search with
`epsilon=0`, and the same PPO horizon used for Dubin (`max_steps=50`). The
script rejects any request to increase that horizon above 50.

Smoke test all three solvers and both players:

```bash
julia --project=experiments \
  experiments/sda/sda-2026-07-18/ppo_solver_response_utilities.jl \
  --test --output-dir /tmp/sda-ppo-response-smoke
```

Full run:

```bash
julia --project=experiments \
  experiments/sda/sda-2026-07-18/ppo_solver_response_utilities.jl \
  --total-timesteps 500000 \
  --eval-runs 500
```

Outputs are `best_response_utilities.csv`, `response_utility_summary.csv`,
`failures.csv` when needed, saved PPO models, and TensorBoard event files under
`ppo_solver_response_utility_results/`. The summed response utility is only an
empirical lower-bound diagnostic; a negative value means PPO underfit and is
uninformative.

## Solver matchup tables

`benchmark_solver_matchups.jl` evaluates each zero-, value-, and full-solver
policy against a uniform opponent, the no-burn heuristic, and that solver's own
PPO response policy. It writes player-specific utility and standard-error tables
under `solver_benchmark_results/` using one shared game-distribution state bank:

```bash
julia --project=experiments \
  experiments/sda/sda-2026-07-18/benchmark_solver_matchups.jl
```

The 18 independent matchup cells run across up to six worker processes by
default, while each cell's rollouts remain serial. Use `--workers N` to choose a
different process count or `--workers 0` for sequential execution. Append
`--test` for a smoke test with two rollouts per cell.

Convert the four result CSVs into a two-panel LaTeX table with:

```bash
julia --project=experiments experiments/solver_benchmark_to_latex.jl \
  experiments/sda/sda-2026-07-18/solver_benchmark_results
```
