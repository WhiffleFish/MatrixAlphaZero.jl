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
