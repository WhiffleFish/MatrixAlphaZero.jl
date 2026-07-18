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
- The default clipped optimizer (`ClipNorm(0.5) + Adam`) replaces the July-10
  SDA script's unclipped Adam optimizer.
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
stronger scripted opponent and run matched zero-value, value-only, and
full-transfer solver response-utility comparisons on the final checkpoint.
