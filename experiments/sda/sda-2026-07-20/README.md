# SDA 2026-07-20: no-transfer training, inference-only fitted priors

This experiment keeps the July-18 SDA task, architecture, optimizer, fixed
evaluation state bank, and 500-query RM+ MCTS configuration. Every training
search starts with zero cumulative regret and strategy, while the value,
average-regret, and average-strategy networks learn from the resulting ordinary
local-search targets.

Evaluation/deployment optionally initializes both fitted priors with
`prior_scale * q_prior(h)`, where `q_prior(h)` is the joint reach of node `h`
under the learned average-policy prior. The default inference scale is 100.
Interpreting `prior_scale = wT1` with the 500-query source solve gives `w = 0.2`,
comfortably below the theoretical ceiling `w = 1`. Training enforces zero prior
scale.
The same reach-weighted mass initializes the floating node count, joint-action
counts under the product policy prior, and the node/edge value averages, so the
critic prior is diluted gradually instead of being replaced by the first sample.

Periodic evaluation benchmarks the transferred solver against a matched
no-transfer solver. They share the same learned value oracle, RM+ configuration,
100-query inference budget, opponents, fixed state bank, and random seeds; only
`prior_scale` differs (`100` versus `0`). Training targets still use fresh
500-query solves, so the comparison tests whether transfer recovers useful play
with one fifth of the deployment search. The learned raw average strategy is
also evaluated separately.

Run from the repository root:

```bash
julia --project=experiments \
  experiments/sda/sda-2026-07-20/train.jl
```

Use `--test` for the standard short smoke run. Override deployment evaluation
strength with `--prior_scale VALUE` or its search budget with
`--inference_tree_queries VALUE`.
