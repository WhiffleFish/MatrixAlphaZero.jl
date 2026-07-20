# Dubin 2026-07-20: no-transfer training, inference-only fitted priors

This experiment keeps the July-16 Dubin architecture and optimizer settings,
but every training search is a fresh, zero-initialized RM+ MCTS solve. The
value, average-regret, and average-strategy networks are still trained from
those ordinary local-search targets.

Regret and average-strategy priors are enabled only in evaluation/deployment.
At node history `h`, both cumulative initializations use
`prior_scale * q_prior(h)`, where `q_prior(h)` is the joint reach under the
learned average-policy prior. The default inference scale is 500, matching the
local search budget. Training enforces a scale of zero.

Run from the repository root:

```bash
julia --project=experiments \
  experiments/dubin/dubin-2026-07-20/train.jl
```

Use `--test` for the standard short smoke run. Override deployment evaluation
strength with `--prior_scale VALUE`.
