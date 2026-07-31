# Network-free Dubin regret transfer

This experiment produces the paper's depth-5 regret-transfer heatmap without a
neural network. It uses the fixed Dubin root reported in the paper, solves the
finite tree by backward induction, and compares ordinary RM+ with RM+ warmed by
imperfect regret and strategy estimates.

Run from the repository root:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl
```

Reference CSV and PDF outputs are in `tabular_regret_transfer_results/`. For a
quick compatibility check:

```sh
julia --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl \
  --test --output /tmp/dubin-transfer-smoke
```
