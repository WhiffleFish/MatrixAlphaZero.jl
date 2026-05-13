Run training first to create regret-matching checkpoints under `regret_matching/`.

Suggested workflow:
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/train.jl`
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/search_llbr.jl`
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/search_llbr_vis.jl`

Quick validation workflow:
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/train.jl --test`
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/search_llbr.jl --test`
- `julia --project=experiments experiments/dubin/dubin-2026-03-27/search_llbr_vis.jl`

Useful `search_llbr.jl` flags:
- `--addprocs`
- `--runs`
- `--checkpoint`
- `--tree_queries`
- `--every`
- `--test`

If `--checkpoint` is omitted, `search_llbr.jl` uses the latest regret-matching
checkpoint.

The experiment trains and evaluates `RegretMatchingSearch()`.

Results are written to `search_llbr_results/<style>/` as CSVs and rendered by
`search_llbr_vis.jl` into a combined PDF/PNG.
