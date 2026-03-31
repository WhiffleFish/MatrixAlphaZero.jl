Run training first to create per-style checkpoints under:
- `matrix_game/`
- `regret_matching/`
- `exp3/`

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

If `--checkpoint` is omitted, `search_llbr.jl` uses the latest checkpoint shared by
all three style directories.

The comparison script evaluates the same Dubin root state under:
- `MatrixGameSearch()`
- `RegretMatchingSearch()`
- `Exp3Search()`

Each style is trained with its own search configuration in `train.jl`, and
`search_llbr.jl` loads the matching checkpoint for each style before running the
LLBR comparison.

Results are written to `search_llbr_results/<style>/` as CSVs and rendered by
`search_llbr_vis.jl` into a combined PDF/PNG.
