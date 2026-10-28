# Dubin 2026-07-20: no-transfer training, inference-only fitted priors

This experiment keeps the July-16 Dubin architecture and optimizer settings,
but every training search is a fresh, zero-initialized RM+ MCTS solve. The
value, average-regret, and average-strategy networks are still trained from
those ordinary local-search targets.

Regret and average-strategy priors are enabled only in evaluation/deployment.
At node history `h`, both cumulative initializations use
`prior_scale * q_prior(h)`, where `q_prior(h)` is the joint reach under the
learned average-policy prior. The default inference scale is 100. Interpreting
`prior_scale = wT1` with the 500-query source solve gives `w = 0.2`, comfortably
below the theoretical ceiling `w = 1`. Training enforces a scale of zero.

Periodic evaluation benchmarks the transferred solver against a matched
no-transfer solver. They share the same learned value oracle, RM+ configuration,
100-query inference budget, opponent, and random seeds; only `prior_scale`
differs (`100` versus `0`). Training targets still use fresh 500-query solves,
so the comparison tests whether transfer recovers useful play with one fifth of
the deployment search.

Run from the repository root:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/train.jl
```

Use `--test` for the standard short smoke run. Override deployment evaluation
strength with `--prior_scale VALUE` or its search budget with
`--inference_tree_queries VALUE`.

## PPO best responses to the paper solvers

`ppo_best_response.jl` approximates best responses to the three solvers in the
paper's round-robin table (`zero_oracle` = SM-MCTS-A, `value_oracle` =
SM-AZ (-RT), `full_solver` = SM-AZ). The solvers are built by
`round_robin_search` with the table's settings: 50 queries, depth 5, search
epsilon 0.1, and regret-only transfer at `prior_scale=2.5`.

For each solver and seat:

1. Behavior-clone the solver's own seat from self-play. The actor is fit to its
   strategy and the critic to its discounted return.
2. Train PPO from that clone (`--ppo-inits bc`, the default; add `scratch` for a
   cold start). PPO's discount defaults to the game's (0.95).
3. Score candidates on 200 selection rollouts. The candidates are:
   - the PPO policy, sampled and greedy;
   - the clone;
   - every solver's seat, including the solver's own;
   - the Dubins heuristic.
4. Report the best candidate on 1000 fresh rollouts, paired with the solver's own
   seat.

Training starts from a 50/50 mixture of the reference state and uniform random
states (`--train-reference-fraction`). Evaluation uses the round robin's fixed
reference state.

The value of any fixed responder lower-bounds the best-response value, so in
`summary.csv`:

- `nashconv_lower_bound` and `paired_gain_sum` both estimate a lower bound on
  NashConv;
- the paired gain over the solver's own seat has the smaller standard error.

`candidates.csv` lists every candidate's selection score. `seats.csv` records
the selected responder, timings, and settings.

A single PPO seed per seat is the default. Run with threads, since PPO
environments, behavior-cloning episodes, and evaluation rollouts are threaded:

```bash
julia -t auto --project=experiments \
  experiments/dubin/dubin-2026-07-20/ppo_best_response.jl
```

Use `--solvers full_solver --players 1` to run a single seat, and `--test` for a
smoke test. Results go to `ppo_best_response_q50_scale2p5/`. Behavior-cloned
warm starts need a CleanRL.jl that accepts `initial_actor` / `initial_critic`.

## PPO response utilities (superseded)

These runs used 100-query solvers with `prior_scale` 100 (or 5) and PPO
`gamma=0.99`, against the game's 0.95. They do not match the paper's tables. PPO
from a cold start also fell below the solvers' own seats against the learned
solvers. `ppo_best_response.jl` replaces this script for paper results.

`ppo_solver_response_utilities.jl` trains one PPO response for each player
against each of three 100-query solvers: a zero oracle, the learned value oracle
without fitted-prior transfer, and the full learned oracle with inference-only
warm starts (`prior_scale=100`). The PPO episode horizon remains capped at 50
steps. Results and TensorBoard event files are written under
`ppo_solver_response_utility_results/`.

Run the complete six-condition evaluation from the repository root:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/ppo_solver_response_utilities.jl \
  --fail-fast true
```

For a quick compatibility smoke test:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/ppo_solver_response_utilities.jl \
  --test
```

After the six PPO response models are available, run the parallel 18-cell
benchmark suite and render its LaTeX table with:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_solver_matchups.jl \
  --workers 6

julia --project=experiments \
  experiments/solver_benchmark_to_latex.jl \
  experiments/dubin/dubin-2026-07-20/solver_benchmark_results
```

## Regret-only deployment transfer

`benchmark_regret_only_transfer.jl` evaluates the component-isolated warm start
introduced after the SDA experiment. The exact cross-domain default is:

- mean-backup RM+ search with 100 queries and depth 5;
- search epsilon 0.1 and action epsilon 0.0;
- raw fitted regret with `prior_scale=5`;
- ordinary learned-policy reach attenuation;
- zero average-strategy and count/value-statistic prior weights.

Against the fixed Dubin heuristics, an independent 1,000-rollout validation
found the scale-5 configuration effectively tied with value-only search:

| search player | value only | regret only, scale 5 | delta |
|---|---:|---:|---:|
| attacker | 0.362 | 0.356 | -0.006 |
| defender | 0.743 | 0.762 | +0.019 |

The direct cross-play result is more favorable. Pooling two independent banks
with 3,000 rollouts per seat, scale 5 has a seat-balanced advantage over
value-only of `0.0285 +/- 0.0064`, with an approximate 95% interval of
`[0.0160, 0.0409]`.

A scale sweep found that Dubin tolerates larger regret mass. Scale 50 pooled to
`0.0328 +/- 0.0063`, but it was worse than scale 5 on the larger independent
2,000-per-seat confirmation bank. The difference between scales 5 and 50 is not
stable enough to justify changing the portable default. Scale 5 is therefore
the recommended setting unless a defender-weighted application is tuned
separately.

Run the heuristic scale screen:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_regret_only_transfer.jl \
  --runs 1000
```

Run value-only versus regret-only cross-play:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/benchmark_value_vs_regret_head_to_head.jl \
  --runs 2000 \
  --scales 5,50
```

The primary outputs are:

- `regret_only_transfer_validation1000/heuristic_matchups.csv`
- `value_vs_regret_head_to_head/matchups.csv`
- `value_vs_regret_head_to_head_confirmation2000/matchups.csv`
- `value_vs_regret_head_to_head_combined/summary.csv`
