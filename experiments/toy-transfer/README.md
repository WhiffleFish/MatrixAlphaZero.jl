# Toy testbed: regret transfer with exact exploitability

A deep-learning-free testbed for the deployment-time regret transfer in
`src/search/mcts.jl`. Everything runs through the production
`MCTSSearch`/`RegretMatchingSearch` code; the only substitutions are a 17-state
tabular "trap game" and a tabular oracle. That makes exploitability **exact**
(backward-induction best response) instead of a PPO response approximation, and
one full mechanism comparison runs in a few minutes.

The point is to have a loop where a transfer mechanism can be selected against
exploitability *and* head-to-head cross-play at the same time, since the SDA and
Dubin evidence disagrees between those two metrics.

## Files

- `toy.jl` — trap game, exact RM+ solver, tabular oracle with two corruption
  models, exact exploitability, and seat-balanced head-to-head with common
  random numbers.
- `calibrate.jl` — measures the SDA regret head's error profile on its held-out
  test split and fits the toy's `NoiseSpec` to it.
- `run.jl` — mechanism comparison across oracle regimes and budgets.

```sh
julia --project=. experiments/toy-transfer/calibrate.jl
julia --project=. experiments/toy-transfer/run.jl --seeds 6 --nreps 64 --episodes 300
```

## The game

Root → six on-distribution "main" subgames (weighted RPS with distinct mixed
equilibria) or, if P2 pays δ = 0.05, a "trap" branch (gate `T` → nine leaf
matrix states `U`). At equilibrium P2 never enters the trap, so the trap branch
is off the self-play distribution while remaining reachable by a best responder.
This is the structure that makes exploitability — a max over best-response
reachable states — diverge from reach-weighted self-play metrics.

## Oracle regimes

| regime | meaning |
|---|---|
| `clean` | exact fitted RM+ artifacts everywhere: an idealized converged network |
| `calibrated` | global regression noise fitted to the measured SDA error profile |
| `calibrated_half` | the same model at half the noise |
| `trap` | localized, confidently wrong prior at the off-distribution trap states: one-hot strategy prior, hallucinated positive regret, and child values that confirm the wrong belief |
| `both` | `trap` on top of `calibrated` |

`calibrate.jl` reports why the `calibrated` regime is the honest default. On the
SDA test split — the *supervised* root distribution, i.e. the best case — the
deployed regret head's induced regret-matching direction closes only **12%
(P1) / 27% (P2)** of the total-variation distance to the true direction that a
uniform prior leaves, at a signal-to-noise ratio of 0.66/0.83. The earlier
hand-picked "confidently wrong one-hot" corruption is a stress test, not a model
of the deployed oracle.

## What the exact exploitability says

Paired by oracle draw against the value-only solver, 6 seeds, 100 queries,
`prior_scale` 5, RM+ with mean backup. Negative Δgap means less exploitable than
value-only; positive h2h means it also wins the cross-play.

| mechanism | Δgap `clean` | Δgap `calibrated` | Δgap `both` |
|---|---:|---:|---:|
| zero oracle | +0.000 | +0.003 | +0.014 |
| `:warmstart` (current) | −0.013 | −0.005 | **−0.017** |
| `:warmstart` + strategy credit | **−0.029** | **−0.012** | +0.012 |
| `:capped` ρ=0.5 | −0.008 | +0.002 | +0.058 |
| `:gated` ρ=0.5 κ=1 | −0.009 | +0.009 | +0.022 |
| `:tempered` ν=0.1 | −0.005 | −0.002 | −0.011 |

Three conclusions, all of which cut against the mechanism this testbed was
originally built to justify:

1. **Where the prior is stored matters, and RM+'s clipping decides it.** In the
   accumulator, `accumulate_regret!(::Plus, …)` clips at zero, so the first
   update that contradicts the injected prior destroys it — the warm start is
   self-limiting. Held in a side accumulator and re-added at every read
   (`:capped`, `:gated`), the same prior is permanent, because the live regret is
   clipped at zero and cannot build the negative counterweight that would cancel
   it. Plain RM does not clip and *can* build that counterweight, which is why a
   visit cap plus evidence gate looked sound when it was developed against the
   default `RegretMatchingSearch()` (`Vanilla`). `--method vanilla` confirms this
   is the cause — in the `both` regime the read-time penalty is 8–58× smaller
   without clipping:

   | mechanism | Δgap, RM+ | Δgap, plain RM |
   |---|---:|---:|
   | `:warmstart` | −0.017 | −0.008 |
   | `:capped` | **+0.058** | −0.000 |
   | `:gated` | **+0.022** | +0.001 |

   So read-time mixing is roughly neutral under plain RM and clearly harmful
   under RM+, which is the variant the deployment uses.
2. **Crediting the average strategy with the transferred mass** — the object
   Lemma "approximate Nash after imperfect transfer" actually bounds, as opposed
   to the deployed `Σₜσₜ/T₂` — helps whenever the strategy prior is reasonable
   and hurts when it is confidently wrong. It is a real inconsistency in the
   deployment, but fixing it is only safe together with a reliability check on
   the strategy head.
3. **Tempering is mildly positive and never harmful here**, but this game is a
   weak test of it: depth 5 covers the whole horizon-3 game, so the toy has very
   few deep, rarely-visited nodes — and those are exactly the population where
   the scale-invariance defect bites in SDA (75% of expanded nodes reach
   `n_s ≤ 2` at a 100-query budget). Prefer the SDA measurements for that claim.

The absolute gaps sit near 0.31 for every mechanism because 100 queries in this
game leaves a large on-distribution search error (`gap_main` ≈ 0.29); the
mechanism differences are the 0.01–0.06 on top of it. Use the paired column, not
the levels.
