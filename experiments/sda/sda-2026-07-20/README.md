# SDA 2026-07-20: no-transfer training, inference-only fitted priors

This experiment keeps the July-18 SDA task, architecture, optimizer, fixed
evaluation state bank, and 500-query RM+ MCTS configuration. Every training
search starts with zero cumulative regret and strategy, while the value,
average-regret, and average-strategy networks learn from the resulting ordinary
local-search targets.

Evaluation/deployment optionally initializes both fitted priors with
`prior_scale * q_prior(h)`, where `q_prior(h)` is the joint reach of node `h`
under the learned average-policy prior. The default inference scale is 500,
matching the local search budget. The matched value-only evaluation uses zero
prior scale, and training enforces zero prior scale.

Run from the repository root:

```bash
julia --project=experiments \
  experiments/sda/sda-2026-07-20/train.jl
```

Use `--test` for the standard short smoke run. Override deployment evaluation
strength with `--prior_scale VALUE`.
