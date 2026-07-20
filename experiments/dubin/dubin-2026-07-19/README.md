# Network-Free Dubin Regret Transfer

This experiment demonstrates regret and average-strategy transfer without a
learned model. It automatically searches for a Dubin initial state where a
short ordinary RM+ search and a short transferred RM+ search have meaningfully
different finite-tree Nash gaps.

The experiment follows the error assumptions in
`sim-az-paper/ideas/current/fitted-error-regret-transfer.tex`:

- leaf-value errors are sampled uniformly from `[-ηV, ηV]`;
- average-regret errors are sampled uniformly from `[-ηR, ηR]`;
- average-strategy errors are sampled uniformly from the intersection of the
  probability simplex and the `L∞` box of radius `ησ`;
- the heatmap uses one transfer-error axis with `ηR = ησ`.

## Tabular source construction

The complete finite Dubin tree and its real continuation values are computed by
backward induction. Every nonterminal node is then treated as the root of an
independent full-remaining-depth search using the repository's `MCTSSearch`
with `RegretMatchingSearch(method=Plus(), backup=:sample)`. These searches are
parallelized across nodes. Matrix-game returns are therefore sampled by the
regret-matching tree search rather than supplied as full-information matrices.

All conditions use one common iteration budget `T`, with `T1 = T2 = T`:

1. Ordinary RM+ runs `T2` local root updates from every node using real leaf
   values.
2. The transfer source runs `T1` local root updates from every node using leaf
   values perturbed uniformly within `[-ηV, ηV]`.
3. Its average regrets and strategies are corrupted by `ηR = ησ`, installed as
   `T1` pseudo-iterations, and a fresh transferred solve runs another `T2`
   local root updates from every node using the real leaf values.

An extra MCTS query is issued internally because the first query only expands
the root; runtime assertions verify that every local solve receives exactly
`T1` or `T2` actual regret updates. Common target-search seeds provide paired
ordinary/transfer comparisons.

The reported finite-tree Nash gap is computed exactly by recursive best
responses over the returned depth-limited policy tree. It is not a claim about
whole-game exploitability.

## Verified default result

With seed `20260719`, depth 5, and `T1 = T2 = 64`, automatic state selection
chose

```text
attacker = (1.806285, 2.411205, -0.270512)
defender = (4.870235, 2.507612, -2.698290)
```

Across 32 paired trials, ordinary RM+ on real leaf values had finite-tree Nash
gap `0.2511 ± 0.0900` (mean and SD). With unperturbed source leaf values and
exact regret/strategy transfer, the second solve reached `0.1514`, for a paired
reduction of `0.0997 ± 0.0177` (mean and SEM). The ordinary reference is the
same for every heatmap cell. Transfer error dominates degradation: at zero leaf
error, transfer error `0.1` removes the mean benefit and larger errors are
harmful. Random bounded leaf error alone has a smaller and non-monotone effect
over this grid, as expected for zero-mean perturbations rather than an
adversarial error construction.

## Run

Use all local Julia threads because the independent all-node searches are
parallelized:

```bash
julia -t auto --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl
```

For a smoke test:

```bash
julia -t auto --project=experiments \
  experiments/dubin/dubin-2026-07-19/tabular_regret_transfer_heatmap.jl --test
```

Important options include `--candidates`, `--finalists`, `--depth`,
`--iterations`, `--trials`, `--leaf-errors`, `--transfer-errors`, `--epsilon`,
`--seed`, and `--output`.

## Outputs

The default `tabular_regret_transfer_results/` directory contains:

- `candidate_states.csv` and `selected_state.csv`;
- `selected_state.png`/`.pdf` and `selected_root_policy.csv`, showing why the
  chosen depth-limited solution is non-myopic;
- `trials.csv` with paired results and achieved errors;
- `summary.csv` with means and SEMs;
- `ordinary_rm_plus_summary.csv` with the single ordinary-RM+ mean, standard
  deviation, SEM, and number of unique evaluations;
- `regret_transfer_heatmaps.png` and `.pdf` with transferred RM+ and paired gap
  reduction panels; the ordinary-RM+ scalar reference is shown above them;
- `config.txt` with the exact run configuration and chosen state.
