# Regret transfer: experiments, failures, and current status

Status: 2026-07-30

## Objective

The goal of regret transfer is to avoid solving every finite-depth local
Markov game from scratch. A network trained on previous local solves should
provide a useful warm start for a new tree, allowing a smaller online search
budget to approach the result of a much larger cold search.

There are three separate questions:

1. Can we learn value, regret, and average-strategy functions from local
   searches?
2. Can those learned quantities accelerate a new local search?
3. Does the accelerated solver remain strong against opponents other than the
   ones used during training or heuristic evaluation?

The first question is mostly a regression problem. The second is what the
heuristic and solver-vs-solver experiments measure. The third is much harder
and is only weakly approximated by the PPO response experiments. We did **not**
have an exact exploitability calculation until the tabular testbed in
`experiments/toy-transfer/`; the SDA and Dubin numbers remain response-based.

## Short version

The original, coupled form of transfer did not work reliably. It transferred
regret, average-strategy mass, visit counts, and value statistics, often with a
large or growing pseudo-mass. This could help early in training, but it became
harmful as the learned priors drifted. Count/value pseudo-observations were
especially damaging because they made inaccurate estimates slow to overwrite.

The current version is much simpler:

- training search is always cold (`prior_scale = 0`);
- regret and average-strategy networks are still trained;
- transfer is used only during inference;
- only nonnegative fitted regret initializes the new tree;
- average-strategy and count/value-statistic pseudo-masses are zero;
- the learned average policy is used only to attenuate regret mass by the
  probability of reaching a node;
- the search uses RM+, mean backup, positive search exploration, and no extra
  exploration when selecting the environment action;
- the transfer mass is deliberately small: normally 5 for 100 search queries,
  or 2.5 for 50 queries.

This regret-only warm start is competitive in heuristic tests and favorable in
direct solver cross-play. It is not a uniform improvement in every player role,
and it has not produced a convincing improvement in the PPO-response
diagnostic.

The 2026-07-30 audit below reaches four conclusions.

- The PPO diagnostic was being read with the wrong sign; the ordering it reports
  already favors regret transfer, and the zero-oracle solver is four times more
  exploitable than either learned solver.
- Scored **per seat**, which is what the Nash characterization requires, regret
  transfer reliably improves player 1's security value (\(+0.83 \pm 0.23\) pooled
  over two state banks) and does nothing for player 2 (\(-0.05 \pm 0.17\)). No
  variant tested, tempering included, closes the player-2 gap.
- The dominant error source is not the transfer at all but the node solver's
  one-sample regret estimator. Replacing it with the exact expectation over the
  matrix the node already stores halves exact exploitability on the toy game.
- The largest single effect in the audit, replicated on two banks, is that the
  learned value function *costs* player 1 security value: the zero-oracle solver
  secures \(+1.15 \pm 0.24\) more as player 1 against a best response than the
  value-guided solver does. The cause is that the critic cannot resolve
  *differences* between sibling actions at the scale the search needs, which is
  invisible to global explained variance — the same metric error made for the
  regret head.

## Transfer operator before 2026-07-30

At an expanded history \(h\), let \(q_{\bar\pi}(h)\) be the joint reach
probability under the learned average-policy prior. The effective transferred
mass is

\[
m_R(h) = \lambda q_{\bar\pi}(h)^\beta,
\]

where the current default is \(\beta=1\). The initial cumulative regret is

\[
R_0(h,a) = m_R(h)\,\widehat{\bar R}(h,a).
\]

Here \(\widehat{\bar R}\) is the network's estimate of regret averaged over
the local source solve. In benchmark names, `raw regret` means this prediction
is used without thresholding, normalization, or another deployment transform;
it does not mean that the regression target is unnormalized cumulative regret.

For RM+, negative fitted regrets are clipped to zero before use. The other
possible warm-start components are disabled:

\[
S_0(h,a)=0,\qquad
n_0(h)=0,\qquad
n_0(h,a)=0,\qquad
\text{return\_sum}_0(h)=0.
\]

The value network is still used normally to evaluate newly expanded frontier
states. What is disabled is treating its prediction as if it were backed by a
large number of previously observed returns.

A representative deployment search is:

```julia
search = AZ.MCTSSearch(
    oracle=learned_oracle,
    tree_queries=100,
    max_depth=5,
    search_style=AZ.RegretMatchingSearch(
        backup=:mean,
        method=AZ.Plus(),
    ),
    value_target=:search,
    ϵ=_ -> 0.1,
    prior_scale=5.0,
    regret_prior_weight=1.0,
    strategy_prior_weight=0.0,
    statistic_prior_weight=0.0,
    prior_reach_power=1.0,
)
```

For the 50-query round robin we use `prior_scale=2.5`, preserving the same
5% prior-mass-to-query ratio.

The distinction between the two epsilons is important:

- **search epsilon:** 0.1, used while traversing the tree;
- **action epsilon:** 0.0, so the action actually executed in the environment
  receives no additional exploration.

Earlier evaluations that set tree-search epsilon to zero do not test the
intended solver and should not be used.

## What we tried

### 1. Training-time SM-OOS transfer

The first implementation transferred information during self-play training as
well as deployment. It included SM-OOS-specific confidence, transfer weight,
and accumulated transfer-time state. The amount of transferred information
could grow with training.

This sometimes helped an early checkpoint, but later checkpoints could
collapse. In the Dubin `honest-glade-29` postmortem, replaying checkpoints with
transfer disabled made both the earlier and final checkpoints roughly
acceptable. Enabling transfer helped the earlier checkpoint and severely hurt
the final checkpoint. The final effective transfer mass was far beyond the
range that replay sweeps found tolerable.

Head ablations also showed that the failure was not simply caused by the value
network. At the final checkpoint, regret transfer reproduced most of the
collapse; strategy transfer alone was less harmful. This showed that a
learned regret prior can become stale or badly calibrated even when it was
useful earlier.

Outcome: **failed as a stable training procedure**. The entire
`SMOOSSearch` implementation/API and outcome-sampling branch were removed.

### 2. Confidence schedules and conservative transfer

We tried varying transfer weight/confidence and using conservative schedules.
Small fixed confidence could survive moderate corruption, whereas larger
confidence became harmful. No single confidence worked reliably across
checkpoints and fit quality.

The main problem was structural: confidence mechanics could reduce the
symptom, but they could not tell whether a learned prior was correct in the
particular local game being solved.

Outcome: **sensitive and not robust enough**. The current inference path has no
learned confidence model or training-time transfer schedule. It exposes a
plain scale parameter instead.

### 3. Separate value, regret, and strategy networks

We tested whether shared-trunk gradient interference explained the poor fits.
The independent Dubin architecture used separate raw-state networks for the
critic, two regret heads, and two strategy heads.

The independent model still had nearly the same high value loss as the
shared-transfer model. This ruled out shared-representation conflict as the
primary explanation, although independent networks remain useful for keeping
the objectives from directly overwriting one another.

Outcome: **did not solve the transfer problem**.

### 4. Correcting the critic-target mismatch

An important incongruity was found between training paths:

- the ordinary no-transfer search used the root search value when
  `value_target=:search`;
- the fitted-regret path historically constructed lambda-GAE targets from
  realized rewards and critic bootstraps, despite logging a similar
  `value_target` setting.

This explained why the transfer runs had much larger critic loss: the runs
were not fitting the same targets. The current `mcts_regret_sim` path honors
`value_target=:search`, and training uses cold local-search values consistently.

Rectifying the mismatch substantially lowered value loss, but the policy and
evaluation curves were nearly unchanged. Better critic fit therefore did not
automatically improve transfer. The transferred regret prior, its calibration,
and the changed search/state distributions remained the relevant bottlenecks.

Outcome: **fixed a real implementation mismatch, but not the performance
problem**.

### 5. No-transfer training, inference-only transfer

We then separated learning from deployment:

- every training tree starts from zero;
- the value, average-regret, and average-strategy networks learn from ordinary
  local RM+ solves;
- transfer is admitted only when constructing a deployment tree;
- the source targets contain no transferred mass, and there is no
  prior-mass subtraction.

The source enforces this invariant:

```julia
iszero(sol.search.prior_scale) || throw(ArgumentError(
    "Training requires MCTSSearch.prior_scale == 0; " *
    "enable fitted priors only for inference",
))
```

This is the foundation of the current experiments. It prevents a bad prior
from changing the targets used to train its successor and makes value-only
versus transfer comparisons much cleaner.

Outcome: **retained**.

### 6. Coupled node warm starts

We tried warm-starting all tree state:

- cumulative regret;
- cumulative average-strategy mass;
- node count \(n(s)\);
- joint-action counts \(n(s,a)\);
- value return sums.

Counts were changed to floating point so pseudo-counts could be represented.
The intended count initialization was approximately

\[
n_0(s)=m\,q_{\bar\pi}(s),\qquad
n_0(s,a)=n_0(s)\bar\pi_1(a_1\mid s)\bar\pi_2(a_2\mid s).
\]

This looked internally consistent but was empirically poor. In the corrected
SDA component ablation, with 100 queries and search epsilon 0.1:

| transferred components | observer reward |
|---|---:|
| cold/value-only search | 19.017 |
| regret only | 20.554 |
| strategy sum only | 17.308 |
| counts and value only | 14.918 |
| regret and strategy | 16.769 |
| all components | 15.936 |

There are two reasons the statistical warm start is particularly dangerous:

1. A value prediction inserted as `return_sum = m * V̂` behaves like \(m\)
   real observations. A large `m` makes a wrong value estimate hard to erase.
2. Marginal average policies do not identify the true joint visitation
   matrix. In general,

   \[
   m\,\bar\pi_1\bar\pi_2^\top
   \ne
   \sum_t \pi_{1,t}\pi_{2,t}^\top.
   \]

Outcome: **failed**. Strategy mass and count/value-statistic mass are disabled
in the selected solver.

### 7. Large transfer scales

The initial inference experiments used scales as large as 100. This could
represent most or all of the online query budget as virtual prior iterations.
With imperfect fits, the new search spent too much of its budget undoing the
prior.

Scale sweeps showed that the best setting is problem-, role-, and
opponent-dependent. Dubin sometimes tolerated scale 50, especially as the
defender, but the advantage over scale 5 did not reproduce consistently on a
larger independent bank. SDA also produced attractive cells at scales 10 or
25, but not in both player roles.

Outcome: **large mass rejected as the portable default**. We use scale 5 with
100 queries and scale 2.5 with 50 queries.

### 8. Sample versus mean backup

`:sample` backs up the sampled trajectory return. `:mean` backs up the running
node mean and is the variant for which the useful convergence guarantees are
available.

The current SDA model was trained from scratch with mean backup. The selected
deployment solver also uses mean backup. The current Dubin deployment
benchmarks use mean backup, but the July 20 Dubin training configuration used
sample backup. That mismatch is a limitation of the Dubin evidence and should
not be hidden.

Outcome: **mean backup retained**, with a remaining matched-training issue for
Dubin.

### 9. More inference queries

Increasing a bad transferred search from 100 to 500 queries did not rescue the
original SDA prior in a compelling way. At 500 queries the cold solver is
already near convergence, so any acceleration advantage is also difficult to
see.

This motivated evaluating 100 and then 50 online queries. With less online
search, a useful warm start has room to matter. The prior scale was reduced
with the query budget rather than leaving its relative influence unchanged.

Outcome: **use smaller deployment budgets to test acceleration**.

### 10. Narrowing the SDA state distribution and increasing capacity

The original SDA initial-state distribution covered a very broad mixture of
orbital regimes. The same state vector could effectively index many different
local games, and the width-32 networks were being asked to generalize across
all of them.

The July 21 SDA experiment:

- restricts target altitude to 600--1200 km;
- restricts observer altitude to within 300 km of the target;
- restricts absolute phase separation to 10--120 degrees;
- uses correlated circular, coplanar initial states;
- increases each independent network from width 32 to width 64;
- uses the same distribution for training and primary evaluation.

This made the learning problem better defined and produced more useful priors.
The old broad distribution remains an out-of-distribution stress test, not the
primary benchmark.

Outcome: **retained**, but it narrows the generalization claim.

### 11. A gated log-magnitude regret regressor

Regret targets are nonnegative, sparse, and highly skewed. We tested a
two-headed hurdle model:

- a BCE-with-logits gate predicts whether each action's regret exceeds
  threshold \(\tau\);
- a second head predicts positive `log1p(regret / tau)` magnitude;
- the magnitude loss is positive-target-only Huber;
- a small reconstruction loss is applied to the final
  gate-times-magnitude prediction;
- inference uses the gate probability continuously.

We also collected a larger frozen final-iterate dataset:

- 65,536 environment decision states;
- one fresh 500-query mean-backup solve per saved root;
- no internal non-root tree states;
- search epsilon 0.1;
- environment action epsilon 0.3 to broaden visited roots;
- trajectory-level train/validation/test splits.

The hurdle model handled nonnegativity and zero structure cleanly, but it did
not give the best overall regret RMSE. On the final test split, the long
softplus single-head refit beat the hurdle model:

| player | long softplus RMSE | hurdle RMSE |
|---|---:|---:|
| 1 | 0.207 | 0.219 |
| 2 | 0.124 | 0.138 |

Outcome: **hurdle architecture not selected**. SDA currently uses the
1000-epoch softplus single-head refits in
`regret_fit_results_softplus_long/models.jld2`.

### 12. Clipping, thresholding, normalization, and reach modulation

The deployment screen tested:

- raw nonnegative regret;
- hard and soft thresholds;
- top-one regret;
- L1 normalization;
- power sharpening;
- gap-scaled normalization;
- clipping;
- reach powers other than one;
- checkpoint versus hurdle/refitted regret sources.

Some transformed variants won individual cells, but the winners changed
between player roles and independent state banks. The simplest raw regret
transfer at low mass was the most stable cross-bank choice.

Outcome: **raw regret with ordinary reach attenuation retained**.

## What currently works

### Against fixed heuristics

At 100 queries and scale 5:

- **Dubin:** regret-only is effectively tied with value-only for the attacker
  and modestly better for the defender. The 1,000-rollout deltas were about
  \(-0.006\) and \(+0.019\), respectively.
- **SDA:** pooling two independent banks, regret-only improved player 1 by
  about \(+3.39\) and player 2 by about \(+0.65\). The player-1 result was much
  clearer statistically; the player-2 result was positive but modest.

These are selected-opponent results, not exploitability results.

### Direct value-only versus regret-only cross-play

When each solver plays both seats against the other:

- Dubin, 100 queries and scale 5:
  seat-balanced regret-only advantage
  \(+0.0285 \pm 0.0064\) over 3,000 rollouts per seat.
- SDA, 100 queries and scale 5:
  seat-balanced regret-only advantage
  \(+1.306 \pm 0.388\) over 500 initial states per orientation.

The 50-query, scale-2.5 three-solver round robins were also favorable:

| problem | full versus value-only, seat-balanced |
|---|---:|
| Dubin | \(+0.052 \pm 0.012\) |
| SDA | \(+0.725 \pm 0.302\) |

In both round robins, full beat value-only and both learned-oracle solvers beat
the zero oracle after averaging over seats. The gains were role-dependent:
Dubin's transfer advantage came mainly from player 2, while SDA's came mainly
from player 1.

Seat-balanced utility for solvers \(A\) and \(B\) is

\[
\bar u(A,B)=
\frac{u(A_{\mathrm{P1}},B_{\mathrm{P2}})
+u(A_{\mathrm{P2}},B_{\mathrm{P1}})}{2}
=
\frac{u(A_{\mathrm{P1}},B_{\mathrm{P2}})
-u(B_{\mathrm{P1}},A_{\mathrm{P2}})}{2}.
\]

It is useful for the strongly role-asymmetric games, but it can conceal which
seat supplies the improvement.

### Against PPO responses

The broader picture is less favorable.

**Score each seat separately.** A Nash profile in a zero-sum game is one where
*each* player's policy is optimal against a best response, so the quantities that
have to improve are the two per-seat best-response utilities:

- as player 1, the value the solver *secures*, \(b_2=\min_{\sigma_2}u(\sigma_1,\sigma_2)\), higher better;
- as player 2, the value it *concedes*, \(b_1=\max_{\sigma_1}u(\sigma_1,\sigma_2)\), lower better.

Their difference is the NashConv. Summing them can hide a change that helps one
seat and hurts the other, and on SDA that is not hypothetical — it is the single
largest effect in the whole audit. Against the fixed exploiter pool, with per-seat
utilities both signed so higher is better for the solver, 300 episodes, paired
under common random numbers:

| solver | seat 1 secures | seat 2 concedes | \(\Delta\) seat 1 | \(\Delta\) seat 2 |
|---|---:|---:|---:|---:|
| zero oracle | \(9.30\) | \(-17.53\) | \(\mathbf{+1.75 \pm 0.32}\) | \(\mathbf{-6.97 \pm 0.47}\) |
| value-only | \(7.55\) | \(-10.56\) | 0 | 0 |

**As player 1 the zero-oracle solver secures more against a best response than the
value-guided solver does**, by \(+1.75 \pm 0.32\) — a five-sigma effect in the
*wrong* direction for the learned value function. It loses overall only because as
player 2 it concedes \(6.97 \pm 0.47\) more. So the intuition that vanilla
SM-MCTS-A does something better here is correct on a per-player basis; it is the
seat-summed NashConv that hides it.

That localizes a real problem in the *learned value function*, not in the transfer:
the critic is fitted to search values generated in self-play, and self-play values
systematically understate what an adversarial player 2 can do to player 1. Player
1's search then trusts continuation values that a best responder refutes. This is
the most promising open thread in the audit and it is a training-side issue —
`value_explained_variance` of 0.79 says the critic fits its targets well, so the
targets are what need to change.

**Read the sign carefully.** `summed_ppo_response_utility` in
`ppo_solver_response_utilities.jl` is
`r1.response_reward + r2.response_reward` with
`response_reward = eval_result.reward[br_player]`, i.e. the sum of the
*responder's own* return across the two seats. In a zero-sum game that equals
\(b_1 - b_2\), the approximate NashConv, so it is the **solver's
exploitability** and **lower is better**. The script's own
`informative_nonnegative = utility_sum >= 0` check, whose warning says a negative
value means "at least one response policy underfit", only makes sense for a
quantity that is nonnegative by construction — which a solver utility would not
be. An earlier revision of this file read the column as a solver utility and drew
the opposite conclusion.

The full SDA table from `best_response_utilities.csv`, decomposed:

| solver | \(b_1\): best response as p1 | \(b_2\): value p1 secures | NashConv \(=b_1-b_2\) |
|---|---:|---:|---:|
| zero oracle (SM-MCTS-A) | 22.907 | 8.220 | \(14.687 \pm 2.12\) |
| value-only | 10.397 | 6.454 | \(3.943 \pm 1.37\) |
| regret-only full solver | 10.399 | 6.760 | \(\mathbf{3.639 \pm 1.32}\) |

So the desired ordering already holds in the PPO diagnostic: full solver \(<\)
value-only \(\ll\) zero oracle, with the zero oracle roughly four times more
exploitable than either learned solver (a gap of 10.7 against a combined standard
error near 2.5). The learned value function is what does the work, halving
\(b_1\) from 22.9 to 10.4.

The transfer-versus-value margin, however, is **not** established: 3.639 against
3.943 is \(0.30 \pm \sim 1.9\). The decomposition localizes it — \(b_1\) is
identical to three decimals (10.399 against 10.397), and the entire difference
sits in \(b_2\) (6.760 against 6.454, each \(\pm 1.1\)), meaning transfer makes
player 1 harder to hold down and does nothing measurable for player 2. That
matches the player-1-dominated transfer gains seen in the SDA cross-play.

The Dubin PPO sums were negative for the learned solvers. A NashConv cannot be
negative, so this is the `informative_nonnegative` check firing: at least one
Dubin response underfit and those numbers carry no information.

Current conclusion: **the ordering holds, the learned value function supplies
almost all of it, and the incremental effect of regret transfer on exploitability
is directionally favorable but inside the noise**.

## 2026-07-30: why the two metrics disagree, and what to change

The head-to-head win and the flat-to-negative response-utility result are not in
tension once you look at what actually reaches the solver. Regret matching
normalizes its input: at a node the played strategy is

\[
\mathrm{RM}\bigl([\widehat{\bar R}]_+\bigr)
=
\frac{[\widehat{\bar R}]_+}{\lVert[\widehat{\bar R}]_+\rVert_1},
\]

so the *magnitude* of the transferred vector is discarded and only its direction
survives. Two consequences follow, and both are measured rather than argued.

### The two safety knobs are inert where the prior decides everything

`prior_scale` and `prior_reach_power` both multiply the transferred vector, so
neither can weaken the prior at a node whose own accumulated regret is still
zero. Sweeping `prior_scale` over four orders of magnitude on the deployed SDA
solver at 100 queries with a fixed RNG stream per state
(`scratch/transfer_probe/probe_scale.jl`, 16 in-distribution roots) confirms this
causally. Mean total-variation distance of the deployed root policy from the cold
solver:

| `prior_scale` | TV, player 1 | TV, player 2 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0.05 | 0.096 | 0.076 |
| 0.5 | 0.102 | 0.074 |
| **5 (deployed)** | **0.107** | **0.136** |
| 50 | 0.146 | 0.219 |
| 500 | 0.269 | 0.165 |

A hundredfold reduction in mass, from the deployed 5 down to 0.05 — worth roughly
\(1/400\) of a single iteration's regret increment — retains 90% (player 1) and
56% (player 2) of the prior's entire effect on the deployed root policy. Almost
all of the transfer's influence is switched on by the prior being nonzero at all,
not by how much mass it carries.

Those zero-regret nodes are the majority. In the same runs a 100-query, depth-5
search expands 91.6 nodes on average, of which **74.6% never exceed \(n_s=2\)**
and 83.9% never exceed \(n_s=4\); the median expanded node is visited once. The
regret and average-strategy heads were supervised **only** at environment
decision states (`includes_nonroot_tree_states = false` in the frozen dataset).
So the warm start is decisive precisely at the nodes it was never fitted on, and
the attenuation designed to express that distrust does not reach the played
strategy.

At the deployed scale the transfer also mildly *sharpens* the emitted root
strategy (mean entropy 0.652 → 0.610 for player 1 and 0.483 → 0.470 for player
2), which is the direction that helps against a fixed opponent and hurts against
a best responder.

### The transferred direction is barely better than uniform, and the model was selected on the wrong metric

`experiments/sda/sda-2026-07-21/score_regret_directions.jl` scores every fitted
regret model by the quantity the search consumes. Write

\[
\mathrm{closed}
=
1 - \frac{\mathbb E\,\mathrm{TV}\bigl(\mathrm{RM}([\widehat{\bar R}]_+),\,\mathrm{RM}([\bar R]_+)\bigr)}
        {\mathbb E\,\mathrm{TV}\bigl(\mathrm{uniform},\,\mathrm{RM}([\bar R]_+)\bigr)},
\]

the fraction of a uniform prior's distance to the true local regret direction
that the fitted direction closes. On the held-out test split — the *supervised*
root distribution, so the best case:

| regret model | rmse p1 | rmse p2 | closed p1 | closed p2 |
|---|---:|---:|---:|---:|
| checkpoint head | 0.223 | 0.148 | 8.6% | 21.0% |
| `softplus_long/baseline` (deployed) | **0.207** | **0.124** | 12.3% | 26.5% |
| `final_iter/baseline` | 0.209 | 0.133 | 15.3% | 31.7% |
| average-strategy head, used as a direction | — | — | 18.2% | 34.2% |
| `hurdle` | 0.219 | 0.138 | 23.4% | 37.0% |
| `hurdle` gate-masked magnitudes | 0.220 | 0.141 | 31.6% | 47.2% |
| baseline magnitudes, gate-masked support | 0.212 | 0.132 | **31.1%** | **48.3%** |

The deployed refit has the best RMSE of every candidate and one of the worst
directions. Validation and test agree on the ranking, so this is not test-split
selection. Three things explain it:

1. \(\bar R\) is an *average* regret and therefore shrinks like
   \(\Delta/\sqrt{T_1}\) as the source solve converges; regressing a vanishing
   residual gives a signal-to-noise ratio near one
   (\(\lVert\bar R\rVert_\infty/\mathrm{RMSE}\) = 0.66 and 0.83).
2. The informative content of an RM+ average-regret vector is largely its
   *support* — which actions are not worth playing. The true support averages
   1.73 of 3 actions; the deployed softplus head emits a strictly positive value
   for all 3 at every state, so it can never express a zero. A squared-error fit
   buys accuracy by smearing mass onto the truly-zero actions, and regret
   matching reads that smear as real probability.
3. RMSE cannot see any of this, because smearing a small amount of mass costs
   almost no squared error while changing the normalized direction a lot.

This retires the §11 conclusion that the hurdle architecture "was not selected"
because its RMSE was worse. Selecting on RMSE inverted the ranking on the metric
that matters.

### Exact exploitability on a toy game rules out read-time prior mixing

`experiments/toy-transfer/` runs the production search on a 17-state tabular trap
game with a tabular oracle, so exploitability is exact and a full mechanism
comparison takes minutes. `calibrate.jl` fits the toy oracle's noise to the
measured SDA error profile above, so the conclusions are not drawn from a
hand-picked corruption. See that directory's README for the table.

The decisive negative result concerns where the prior is stored, and it turns on
RM+'s clipping:

- **In the accumulator** (`:warmstart`, `:tempered`): `accumulate_regret!(::Plus,
  …)` clips at zero, so the first update that contradicts the injected prior
  destroys it. The warm start is self-limiting.
- **Outside the accumulator, re-added at read time** (`:capped`, `:gated`): the
  prior is permanent. The live regret is clipped at zero and therefore cannot
  build the negative counterweight that would cancel a term re-injected at every
  visit. Plain RM, which does not clip, *can* build that counterweight — which is
  why a visit cap plus an evidence gate looked sound when it was first developed
  against the default `RegretMatchingSearch()`, i.e. `Vanilla`.

Rerunning the same comparison with `--method vanilla` confirms clipping is the
cause. Paired \(\Delta\)gap against value-only, corrupted regime:

| mechanism | RM+ | plain RM |
|---|---:|---:|
| `:warmstart` | −0.017 | −0.008 |
| `:capped` | **+0.058** | −0.000 |
| `:gated` | **+0.022** | +0.001 |

Read-time mixing is roughly neutral without clipping and clearly harmful with it,
which is the variant the deployment uses. Any usage-side fix must therefore keep
the prior inside the regret accumulator, which is what `:tempered` does.

### The deployed average strategy is not the object the theory bounds

The paper's Lemma "approximate Nash equilibrium after imperfect transfer"
(`lem:approx-nash` in `theorems.tex`) bounds
\((wT_1\widehat{\bar\sigma} + \sum_t\sigma_t)/(wT_1+T_2)\). The deployment sets
`strategy_prior_weight = 0` and reports \(\sum_t\sigma_t/T_2\), so the prior's
mass is charged to the regret recursion but never credited to the average. In the
toy this inconsistency is worth about −0.03 gap when the strategy head is decent
and +0.01 when it is confidently wrong, so repairing it is only safe alongside a
reliability check on the strategy head. It is available as
`strategy_prior_weight = 1`.

### Selected change: uniform-tempered warm start

`transfer_mode = :tempered` redistributes the transferred vector toward uniform
**at fixed total mass**:

\[
R_0(h,\cdot) = m_R(h)\Bigl[\lambda(h)\,w(h,\cdot) + \bigl(1-\lambda(h)\bigr)\tfrac{\lVert w\rVert_1}{|\mathcal A|}\Bigr],
\qquad
w = [\widehat{\bar R}(h,\cdot)]_+,
\]
\[
\lambda(h) = \frac{q_{\bar\pi}(h)^\beta\lVert w\rVert_1}
                  {q_{\bar\pi}(h)^\beta\lVert w\rVert_1 + \nu\,\widehat\Delta(h)},
\]

with \(\widehat\Delta(h)\) the payoff range of the node's own matrix game. The
node then plays the explicit mixture
\((1-\lambda)\cdot\text{uniform} + \lambda\cdot\mathrm{RM}(w)\), which is an
improvement over the previous operator in five specific ways:

1. \(\lambda\) — how hard a fresh node commits to the prior — becomes a
   controlled quantity instead of an artifact of the residual's shape.
2. \(\lambda\) falls with the reach, so `prior_reach_power` finally affects the
   played strategy and not just the mass.
3. \(\lambda\) rises with \(\lVert w\rVert_1\) measured against the node's own
   payoff range. Since \(\bar R\to0\) as the source converges, a fitted magnitude
   small next to the local payoff range is evidence that the direction is mostly
   fitting error, and the node should stay near uniform. Dividing by
   \(\widehat\Delta(h)\) keeps this scale-free across SDA states whose SNR
   magnitudes differ by orders of magnitude.
4. Total mass is preserved, so \(\lVert R_0\rVert_1\) is exactly the
   \(wT_1\widehat{\bar R}\) mass the theorem prescribes, and
   \(\Phi(R_0)\le\Phi(m_R w)\) because moving mass toward the mean cannot raise a
   sum of squares. Whenever the untempered warm start satisfied
   the theorem's weight condition, the tempered one satisfies it a fortiori — nothing in
   the theory has to be re-derived.
5. \(\nu=0\) reproduces the current warm start exactly, and \(\nu\to\infty\)
   leaves a uniform vector of the same small mass that a single real iteration
   overwhelms, i.e. the cold value-only solver. Both baselines are endpoints of
   one continuous knob, which is what makes "at worst does not harm" structural
   rather than empirical: a prior that turns out to be bad on a new domain can be
   dialed out without touching anything else.

The tempering is added to the accumulator, so RM+ clipping still erodes it.

### Diagnostic knob

`transfer_max_depth` restricts the warm start to depths at or above the root.
Setting it to 0 confines the transfer to the states the heads were actually
fitted on and isolates how much of the measured transfer effect comes from
applying the prior off its training support.

### Training-run audit (`giddy-waterfall-52`, wandb)

The training logs already contained the fit diagnosis. Final values, 1221
iterations, SDA 2026-07-21:

| metric | value | reading |
|---|---:|---|
| `oracle_quality/target_regret_l2` | 0.152 | mean \(\lVert\bar R\rVert_2\) per player |
| `oracle_quality/regret_pred_mse` | 0.0225 | per-component MSE, so \(\lVert\text{err}\rVert_2 \approx 0.150\sqrt3 = 0.260\) |
| `oracle_quality/target_policy_kl_p1` / `_p2` | 0.91 / 0.75 | flat all run; the 3-action maximum is \(\log 3 = 1.10\) |
| `oracle_quality/policy_kl_p1` / `_p2` | 0.0005 | checkpoint-to-checkpoint, i.e. the head is stable |
| `oracle_quality/value_explained_variance` | 0.79 | the critic fits |

So the regret head's prediction error is about **1.7 times the size of the regret
signal itself** on the training distribution at the end of training, which
independently reproduces the held-out signal-to-noise measurement. The strategy
head sits about 0.9 nats from its targets and never improves, while being stable
across checkpoints — it is not diverging, it simply cannot represent the local
solve's average strategy. The value head is the only one that fits, which is
consistent with the learned value function supplying nearly all of the measured
benefit.

One methodological note on the run's own monitoring: its config records
`inference/prior_scale = 100` and
`inference/prior_components = regret_strategy_counts_value`. The periodic
`transfer_search` eval arms therefore measured the fully coupled warm start at
scale 100 — a configuration later rejected on both axes (the component ablation in
§6 put all-components at 19.017 \(\to\) 15.936, and §7 rejected large scales). The
resulting curves (`transfer_search_observer_vs_no_burn` 13.9 against
`no_transfer_search_observer_vs_no_burn` 19.1) are not evidence about the deployed
scale-5 regret-only solver, and the run never monitored the configuration that was
eventually shipped.

Query the run without the wandb SDK's `wandb-core` subprocess using
`scratch/transfer_probe/wb_api.py`, a small GraphQL client.

### The node regret estimator, not the transfer, is the dominant error source

`update_node!` formed each player's instantaneous regret from the single opponent
action sampled on that visit,

\[
\Delta_1 = q[:,j] - \mathrm{total},\qquad \Delta_1[i] := 0,
\]

which is a one-sample estimate. Since roughly three quarters of expanded nodes
never exceed \(n_s=2\), most nodes never average that sample down. It also mixes
baselines: off-diagonal entries are compared against the sampled `total`, the
played action is forced to zero, and `total` uses the freshly returned child value
while \(q\) uses the child's running mean.

The full matrix \(q=r+\gamma v\) is already stored at every expanded node, so the
exact expectation under the node's own strategy pair costs one matrix-vector
product per player and no oracle call:

\[
\Delta_1 = q\sigma_2 - \sigma_1^\top q\sigma_2,\qquad
\Delta_2 = \sigma_1^\top q\sigma_2 - q^\top\sigma_1 .
\]

This is ordinary regret matching on the node's estimated matrix game, available
as `RegretMatchingSearch(update=:expected)`. It uses strictly the same
information — the sampled variant already reads an entire column of \(q\),
including entries no simulation has refined — so the trade is explicit: it removes
the opponent-sampling variance and the baseline inconsistency in exchange for
weighting every entry of \(q\), including unrefined ones. It also makes the node's
regret independent of the \(\epsilon\)-exploration, which then only decides where
child values get refined.

Exact exploitability on the toy, 6 oracle seeds, 100 queries, RM+ mean backup:

| regime | value-only, `:sampled` | value-only, `:expected` |
|---|---:|---:|
| clean | 0.320 | \(\mathbf{0.161}\) |
| calibrated | 0.312 | \(\mathbf{0.161}\) |
| both | 0.310 | \(\mathbf{0.146}\) |

Every solver's exploitability roughly halves, an order of magnitude more than any
transfer mechanism moved it, and the search is about 20% *faster* (4.39 against
5.35 ms per SDA search) because the allocating slice-and-patch is replaced by a
matrix-vector product.

It also changes what the warm start is worth. Paired \(\Delta\)gap for
`warmstart` against value-only:

| regime | `:sampled` | `:expected` |
|---|---:|---:|
| clean | \(-0.0131 \pm 0.0044\) | \(\mathbf{-0.0395 \pm 0.0021}\) |
| both (corrupted) | \(-0.0169 \pm 0.0035\) | \(-0.0009 \pm 0.0032\) |

Three times the benefit with a good prior, and exactly neutral with a corrupted
one. The mechanism is mechanical rather than mysterious: a warm start perturbs the
*initial* cumulative regret, so its influence is measured against the
per-iteration increments. When those increments are one-sample estimates whose
noise is comparable to their mean, the prior's information is swamped and whether
it helps is close to a coin flip. Reducing the increment variance is what lets the
prior's signal survive.

Two caveats. `:expected` also changes `fresh_regret`, hence the emitted training
targets, so a model trained under `:sampled` is deployed off-distribution when the
estimator changes — the SDA numbers below are inference-only and the full benefit
plausibly needs a retrain. And the SM-MCTS-A composition argument that licenses
the per-node guarantee assumes the sampled-return estimator, so that argument
needs re-checking rather than being assumed to carry over.

### SDA measurements so far, and where this was paused

Fixed exploiter pool and seat-balanced cross-play, 150 episodes, 50 steps, 100
queries, `prior_scale` 5, checkpoint 1221. Pool numbers are signed from the
solver's perspective, so **higher is better** here (the opposite convention from
`summed_ppo_response_utility`). Every solver faces the same three PPO responses
and the same initial states under common random numbers.

Deployed `:sampled` node solver, 100 queries, `prior_scale` 5, 300 episodes per
state bank, paired against value-only under common random numbers. Both seat
columns are signed so higher is better for the solver. **Two independent state
banks** (seeds 20260730 and 909090), because §12 already recorded that transfer
winners move between banks — and they do again here.

| quantity | bank 1 | bank 2 | pooled |
|---|---:|---:|---:|
| zero oracle, seat 1 | \(+1.006 \pm 0.352\) | \(+1.284 \pm 0.319\) | \(\mathbf{+1.145 \pm 0.238}\) |
| zero oracle, seat 2 | \(-6.957 \pm 0.450\) | \(-7.398 \pm 0.483\) | \(\mathbf{-7.177 \pm 0.330}\) |
| `warmstart`, seat 1 | \(+0.858 \pm 0.314\) | \(+0.795 \pm 0.337\) | \(\mathbf{+0.827 \pm 0.230}\) |
| `warmstart`, seat 2 | \(+0.021 \pm 0.225\) | \(-0.111 \pm 0.240\) | \(-0.045 \pm 0.165\) |
| `temper0p03`, seat 1 | \(+0.830 \pm 0.319\) | \(+0.409 \pm 0.296\) | \(+0.620 \pm 0.218\) |
| `temper0p03`, seat 2 | \(+0.323 \pm 0.223\) | \(-0.180 \pm 0.243\) | \(+0.071 \pm 0.165\) |

What replicates, and what does not.

1. **Regret transfer reliably improves player 1's security value and does nothing
   for player 2.** Pooled, `warmstart` gains \(+0.827 \pm 0.230\) on seat 1 (3.6
   sigma) and \(-0.045 \pm 0.165\) on seat 2. Seat 2's sign flips between banks
   for every variant tested. On the per-seat Nash criterion the transfer therefore
   satisfies "does not harm" on both seats and "improves" on one only.
2. **Tempering does not close the seat-2 gap.** On bank 1, `temper0p03` looked
   like the answer: seat 2 \(+0.323 \pm 0.223\), the largest summed gain of seven
   variants, the best pooled worst case. None of that survived bank 2, where its
   seat 2 is \(-0.180 \pm 0.243\) and its summed gain drops from \(+1.154\) to
   \(+0.229\). Pooled it is indistinguishable from `warmstart` on both seats and
   slightly worse on seat 1. An earlier revision of this file recommended it on
   single-bank evidence; that is withdrawn. `temper0p1` was the best variant on
   bank 2 and two banks cannot resolve the tempering weight.
3. **The zero oracle's seat asymmetry replicates and is the largest effect in the
   audit.** As player 1 the zero-oracle solver secures \(+1.145 \pm 0.238\)
   (4.8 sigma) more against a best response than the value-guided solver, on both
   banks independently and again under the `:expected` node solver
   (\(+1.747 \pm 0.317\)). It loses overall only through its player-2 exposure,
   \(-7.177 \pm 0.330\). So the learned value function *costs* player 1 security
   value, and the seat-summed NashConv hides it.

   **The cause is differential resolution, not self-play optimism.** An earlier
   revision of this file attributed it to self-play value targets being optimistic
   for player 1. That explanation is wrong on its own terms: adding a constant to
   every entry of a zero-sum matrix game changes its value but not its
   equilibrium, so a uniform optimism bias cancels exactly and cannot move the
   decision. What matters is the *centered* error across a node's own children.

   Measured directly at SDA root matrices (`scratch/transfer_probe/probe_value_scale.jl`,
   40 states, reference = 600-query depth-8 search):

   | quantity | value |
   |---|---:|
   | spread of immediate reward across the 9 joint actions | \(0.000\) |
   | spread of \(\gamma\widehat V\) across the 9 children | \(0.743\) |
   | spread of the reference matrix \(q_{\mathrm{ref}}\) | \(4.214\) |
   | centered RMSE of \(\gamma\widehat V\) (the decision-relevant error) | \(1.377\) |

   Three facts compose. One SDA step changes the immediate reward by *exactly
   zero* — orbital geometry takes many steps to respond — so every bit of
   discrimination at a node must come from the frontier value. The learned value's
   spread across siblings (0.743) is only 18% of the true spread (4.214), so it is
   badly too flat. And its centered error (1.377) is nearly twice its own spread,
   so what variation it does have is mostly error.

   The consequence is measurable as a per-node decision regret. Playing the
   equilibrium of each approximate matrix and scoring what player 1 actually
   secures on \(q_{\mathrm{ref}}\):

   | frontier used | p1 secures | regret vs oracle |
   |---|---:|---:|
   | \(q_{\mathrm{ref}}\) (oracle) | \(15.360\) | 0 |
   | \(r + \gamma\widehat V\) (learned) | \(13.808\) | \(1.552\) |
   | \(r\) alone (zero frontier) | \(14.466\) | \(0.894\) |

   paired difference \(+0.658 \pm 0.177\) (3.7 sigma) against the learned value.
   This reproduces the whole-search seat-1 effect at a single node, with no search
   dynamics and no exploiter pool involved. Note that with zero reward spread the
   zero-frontier matrix is constant, so its "equilibrium" is uniform — a uniform
   strategy secures more than the one the learned value implies.

   Caveat: this probe is a one-step caricature of a depth-5 search, which does
   accumulate real reward spread over five steps and therefore leans less on
   \(\widehat V\) than the probe suggests. It identifies the mechanism, not its
   full magnitude.

   The actionable consequence is a metric error, and it is the same one made for
   the regret head. `value_explained_variance` of 0.79 is computed over the whole
   state distribution, where \(\widehat V\) varies by about \(\pm 7.6\);
   explaining 79% of that leaves an RMSE near 3.5, which is respectable globally
   and useless for ranking siblings whose true spread is 4.2. **A value function
   used inside a search should be scored on sibling-differential accuracy, not on
   global explained variance**, and trained with a loss that weights those
   differences.

Single-bank results, not yet replicated and not to be relied on: `depth0` (root-
only transfer) showing a negative seat-2 delta and losing cross-play, and the
per-variant cross-play ordering. The support-masked prior's failure has two
independent supports — worse seat 1 on bank 1, and seat 2 \(-0.344 \pm 0.182\)
under `:expected` while its summed delta reads a neutral \(-0.038 \pm 0.384\) —
so that one is retained as a finding.

To reproduce, or to extend to the remaining variants:

```bash
julia --project=experiments experiments/sda/sda-2026-07-21/benchmark_transfer_modes.jl \
    --episodes 300 --max-steps 50 \
    --solvers zero_oracle,value_oracle,warmstart,masked_warmstart,temper0p03,masked_temper0p03,depth0 \
    --output experiments/sda/sda-2026-07-21/transfer_mode_benchmark_perseat
# add --update expected for the exact-expectation node solver
```

Run it as a single sequential process. Sharding it across three processes on a
four-core machine was about four times slower in wall clock than running the
solvers one after another. Roughly 180 s per pool cell for a value-only solver
and 300 s for a transfer solver at 150 episodes, scaling linearly in episodes.
The transfer solvers are slower because `SupportMaskedRegretOracle` and the
`FittedRegretModel` evaluate the regret and strategy heads at every expanded
node, while the value-only and zero oracles return constants.

Open items, in the order that would move the transfer-versus-value margin:

1. Finish the variant sweep above, and sweep `transfer_temper` — with the reach
   exponent active, \(\nu = 0.1\) probably tempers everything below depth 1 nearly
   to uniform, so smaller \(\nu\), or the reach-free `temperflat` variants, are
   likely the useful settings. `scratch/transfer_probe/probe_temper.jl` reports
   where each \(\nu\) sits between the warm start and the cold solver; it was
   written but never run.
2. Train PPO responses *against* the masked and tempered solvers, with more than
   one seed, so the \(0.30 \pm 1.9\) margin becomes decisive. Needs the
   `experiments` environment, whose Wandb/PythonCall dependency requires a Conda
   bootstrap with network access.
3. Repeat the direction scoring and the tempering sweep on Dubin. Everything
   dated 2026-07-30 is SDA-only.
4. Regenerate the frozen regret dataset with internal tree states. The prior is
   decisive at nodes it was never fitted on, and no deployment-side mechanism can
   manufacture signal that the fit does not contain. Largest expected gain and
   largest cost.
5. Consider retiring `:capped`/`:gated` from `src/search` once their ablation
   role is written up; they exist only to record that read-time mixing fails
   under RM+.

## The learned average-strategy network

The average-strategy network is still trained because it is a useful target
and supplies the learned reach \(q_{\bar\pi}(h)\). However:

- its standalone policy has generally been weaker than value-guided search;
- directly transferring its cumulative strategy mass was harmful in the
  component ablations;
- marginal policy heads cannot reconstruct exact joint visitation counts;
- inference applies it at internal tree states that are less directly
  supervised than environment trajectory roots.

The current solver therefore uses the strategy network only to attenuate
regret mass by node reach. It does not add the strategy prediction to
`policy_sum`.

## Residual evidence

Approximate Shapley/fixed-point residual sweeps measure

```text
search backup - current value prediction
```

over checkpoints. The completed on-policy sweeps showed improving bulk error:

- Dubin late L1/L2 residuals were about 33%/36% lower than early residuals;
- SDA late L1/L2 residuals were about 37%/21% lower.

SDA's worst-case residual was flat or slightly worse and its root residual
improved much less, indicating persistent hard states. These sweeps used
freshly sampled states at each checkpoint, so they mix function improvement
with a changing visitation distribution. A fixed state-bank sweep is still
the cleaner test for a paper-quality convergence claim.

## What remains unresolved

1. **Internal-node distribution shift.** Regret fits are supervised only at
   environment search roots, while deployment initializes every expanded
   internal node. Reach attenuation was the intended mitigation but, before
   tempering, could not reach the played strategy at all (2026-07-30 section).
   Supervising the heads at internal tree states is the untried fix.
2. **Calibration.** A low global scale works, but it is not a calibrated
   estimate of local prior reliability. Tempering makes the *shape* respond to
   ‖R̄̂‖₁/Δ̂(h) and to reach, which is a local reliability proxy, not a learned
   confidence model.
3. **Role asymmetry.** The selected transfer wins come from different roles in
   Dubin and SDA. A single pooled number hides this.
4. **Average-policy reach quality.** The current reach product is useful as a
   heuristic but is not a learned joint occupancy model.
5. **Dubin backup mismatch.** The selected Dubin deployment uses mean backup,
   whereas its July 20 checkpoint was generated with sample-backup training.
6. **Out-of-distribution behavior.** The SDA result is in the restricted LEO
   regime. Performance over the old broad orbital mixture is not established.
7. **General robustness.** Heuristic and solver cross-play wins have not
   translated into a better PPO-response aggregate.
8. **Exact exploitability.** No SDA or Dubin experiment computes it.
   `experiments/toy-transfer/` now does, on a 17-state tabular game, which is
   what settled the read-time-mixing question; the SDA and Dubin numbers remain
   response-based. An exact truncated best response was tried for SDA and is
   degenerate at feasible horizons (orbital geometry gives zero reward
   differential over 2--4 steps), so the fixed exploiter pool in
   `benchmark_transfer_modes.jl` is the affordable substitute.
9. **PPO responses for the new variants.** The exploiter pool reuses responses
   trained against the three original solvers. Retraining a PPO response
   *against* the tempered/support-masked solver is still outstanding; it needs
   the `experiments` environment with its Conda-backed Wandb dependency.
10. **Dubin.** Every 2026-07-30 measurement is SDA-only. The direction-quality
   score and the tempering sweep both need repeating on the Dubin heads.

## Practical recommendation

For current experiments, use:

- no-transfer training;
- search-value critic targets;
- RM+ with mean backup;
- 500-query training solves;
- 50 or 100 inference queries;
- search epsilon 0.1 and action epsilon 0;
- learned value frontier evaluations;
- nonnegative regret-only transfer on the **raw** fitted regret;
- `transfer_mode=:warmstart` remains the default. `:tempered` is available and is
  the better-motivated operator — it is the only form in which `prior_scale` and
  the reach attenuation affect the played strategy at all — but on two independent
  SDA state banks it is not measurably better than `:warmstart` on either seat, so
  there is no evidence-backed reason to switch yet. If you do use it, sweep
  `transfer_temper` across at least two banks first;
- scale 2.5 at 50 queries or 5 at 100 queries;
- no transferred strategy mass, unless the strategy head's reliability has been
  checked — the theory-consistent `strategy_prior_weight=1` helps only then;
- no count or value-statistic pseudo-mass;
- the restricted correlated LEO distribution for primary SDA claims.

Consider also `RegretMatchingSearch(update=:expected)`, which halves exact
exploitability on the toy and improves both seats for the no-transfer solvers on
SDA, but changes the emitted regret targets and so wants a retrain before it is
adopted as the default.

Do not use:

- `transfer_mode=:capped` or `:gated`. Read-time prior mixing defeats RM+'s
  clipping, which is the solver's built-in mechanism for forgetting a refuted
  warm start, and measurably raises exact exploitability on the toy game.
- the support-masked regret prior. It improves the fitted direction and wins
  cross-play but does not improve per-seat best-response utility, and under the
  `:expected` node solver it makes seat 2 worse.
- `transfer_max_depth` as a remedy. Restricting the prior to the states it was
  fitted on is worse than applying it everywhere.
- regression RMSE as the regret-model selection criterion. It anti-correlates
  with the induced regret-matching direction; score candidates with
  `score_regret_directions.jl`. Note, though, that a better direction did not
  translate into better best-response utility, so treat that score as a fit
  diagnostic and not as a deployment criterion.

Always report:

1. value-only and zero-oracle controls;
2. **per-seat best-response utility, never only the seat-summed NashConv.** A
   Nash profile requires each player's policy to be optimal against a best
   response, so both seat deltas must be nonnegative; the sum cancels a change
   that helps one seat and hurts the other, which is exactly what the support-
   masked prior does;
3. both player roles separately;
4. seat-balanced direct cross-play as a secondary summary;
5. heuristic results as selected-opponent performance;
6. PPO results as empirical response diagnostics, not exact exploitability;
7. the fitted prior's `closed` score, so a transfer claim is always accompanied
   by how much signal the transferred direction actually carries.

## Main code and result locations

- Core warm start and transfer modes: `src/search/mcts.jl`
  (`warmstart_node!`, `temper_transfer_regret!`, `transfer_mass`)
- Direction-quality scoring of regret models:
  `experiments/sda/sda-2026-07-21/score_regret_directions.jl`
- SDA transfer-mode comparison (exploiter pool + cross-play):
  `experiments/sda/sda-2026-07-21/benchmark_transfer_modes.jl`
- Exact-exploitability toy testbed: `experiments/toy-transfer/`
- `prior_scale` inertness probe: `scratch/transfer_probe/probe_scale.jl`
- Search configuration: `src/search/api.jl`
- Training-time no-transfer guard: `src/solver.jl`
- Dubin experiment:
  `experiments/dubin/dubin-2026-07-20/`
- SDA experiment:
  `experiments/sda/sda-2026-07-21/`
- SDA regret fitting:
  `experiments/sda/sda-2026-07-21/fit_regret_hurdle.jl`
- Transfer transformation screen:
  `experiments/sda/sda-2026-07-21/benchmark_regret_transfer_schemes.jl`
- Dubin 50-query round robin:
  `experiments/dubin/dubin-2026-07-20/solver_round_robin_q50_scale2p5/`
- SDA 50-query round robin:
  `experiments/sda/sda-2026-07-21/solver_round_robin_q50_scale2p5/`
- Approximate residual diagnostic:
  `experiments/approximate_shapley_residual.jl`
