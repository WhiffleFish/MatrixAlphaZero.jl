# W&B Logging

This repo logs to Weights & Biases through `ExperimentTools.WandbCallback`.
The shared callback accepts a callback `NamedTuple`, keeps finite numeric fields,
maps known fields into stable metric namespaces, and logs them with
`step = info.iter`.

W&B logging is enabled by the Dubin training scripts only when
`WANDB_API_KEY` is present in the environment. Current Dubin runs use the
project `Matrix AlphaZero`.

## Metric Namespaces

Known metric fields are grouped by `ExperimentTools/src/wandb_callback.jl`.
Any finite numeric field that is not in one of these groups is still logged
under its raw key. This is how the slash-prefixed eval metrics are emitted.

The callback deliberately skips nonnumeric values and nonfinite values. It also
skips oracle objects (`oracle`, `online_oracle`, and `ema_oracle`). Per-minibatch
tables are not logged.

## Progress

These metrics describe solver progress and are emitted by the core solver.

| Metric | Meaning | Notes |
|---|---|---|
| `progress/iter` | Solver callback iteration. | Used as the W&B step. Initial callback logs `0`. |
| `progress/update` | Training update count. | Same value as `iter` in normal solver callbacks. |
| `progress/steps_done` | Total self-play samples accumulated so far. | Starts at `0`. |
| `progress/max_steps` | Solver target sample budget. | Constant for a run. |
| `progress/sim_depth` | Simulation depth used for self-play. | Constant for a run unless the solver config changes. |
| `progress/samples_added` | Number of samples added by the latest self-play batch. | Not present on the initial callback. |
| `progress/exploration_epsilon` | Search exploration epsilon for the current update. | Comes from the search epsilon schedule. |
| `progress/transfer_tau` | Current SM-OOS transfer temperature/state. | SM-OOS only. MCTS/regret-matching runs do not emit this. |

## Self-Play

These metrics summarize the trajectories generated in the latest self-play
batch.

| Metric | Meaning |
|---|---|
| `selfplay/mean_ep_length` | Mean episode length across generated trajectories. |
| `selfplay/mean_reward` | Mean reward across generated trajectory rewards. |
| `selfplay/reward_std` | Standard deviation of generated trajectory rewards. |
| `selfplay/mean_search_time` | Mean per-search runtime from the generated histories. |
| `selfplay/total_search_time` | Total search runtime from the generated histories. |
| `selfplay/search_count` | Number of search calls recorded in the generated histories. |
| `selfplay/batch_size` | Number of samples in the merged training batch. |

## Training Health

These metrics summarize optimizer minibatches from the latest training update.

| Metric | Meaning | Modes |
|---|---|---|
| `training_health/mean_loss` | Mean total training loss. | All modes. |
| `training_health/mean_value_loss` | Mean critic/value loss. | All modes. |
| `training_health/learning_rate` | Learning rate used for the current optimizer update after decay and clipping. | All modes. |
| `training_health/mean_grad_norm` | Mean gradient norm across minibatches. | All modes. |
| `training_health/max_grad_norm` | Maximum gradient norm across minibatches. | All modes. |
| `training_health/mean_regret_loss` | Mean regret-head loss. | `FittedRegretModel` / SM-OOS. |
| `training_health/mean_strategy_loss` | Mean strategy-head loss. | `FittedRegretModel` / SM-OOS. |
| `training_health/mean_policy_loss` | Mean actor policy loss. | `ActorCritic` / MCTS regret-matching. |

The solver still constructs `minibatch_metrics` for local callbacks, but
`WandbCallback` does not log it. This avoids a large W&B table that is hard to
read and not useful for run comparison.

## Oracle Quality

These metrics compare the current callback oracle against the previous callback
oracle and against the latest training batch. `oracle_metrics` samples up to
128 batch elements, so these are diagnostics rather than full-batch exact
statistics.

| Metric | Meaning | Modes |
|---|---|---|
| `oracle_quality/value_pred_mse` | Mean squared error between value predictions and sampled value targets. | All modes. |
| `oracle_quality/value_explained_variance` | `1 - Var(target - prediction) / Var(target)` for sampled value targets. | All modes. Skipped when sampled targets have zero variance. |
| `oracle_quality/policy_entropy_p1` | Mean entropy of player 1 target policy/strategy distributions. | All modes. |
| `oracle_quality/policy_entropy_p2` | Mean entropy of player 2 target policy/strategy distributions. | All modes. |
| `oracle_quality/policy_kl_p1` | KL divergence from current player 1 policy to previous callback player 1 policy. | All modes. |
| `oracle_quality/policy_kl_p2` | KL divergence from current player 2 policy to previous callback player 2 policy. | All modes. |
| `oracle_quality/target_policy_kl_p1` | KL divergence from current player 1 policy to sampled player 1 target policy/strategy. | All modes. |
| `oracle_quality/target_policy_kl_p2` | KL divergence from current player 2 policy to sampled player 2 target policy/strategy. | All modes. |
| `oracle_quality/target_regret_l2` | Average L2 norm of sampled regret targets. | `FittedRegretModel` / SM-OOS. |
| `oracle_quality/regret_pred_mse` | Mean squared error between predicted regrets and sampled regret targets. | `FittedRegretModel` / SM-OOS. |

The W&B key whitelist also contains older `strategy_*` names, but the current
`oracle_metrics` implementation emits the `policy_*` names above for both
strategy-target and policy-target training modes.

## Heuristic Evaluation

The current Dubin experiment scripts run periodic evaluation rollouts against
the hand-coded heuristic policies. These metrics are logged only on iterations
where `iter % eval_every == 0`.

There are two matchups:

| Prefix | Meaning |
|---|---|
| `eval/az_p1_vs_heuristic/*` | AlphaZero controls player 1, the Dubin defender heuristic controls player 2. |
| `eval/heuristic_vs_az_p2/*` | The Dubin attacker heuristic controls player 1, AlphaZero controls player 2. |

Each matchup logs:

| Metric suffix | Meaning |
|---|---|
| `reward` | Mean reward for the AlphaZero-controlled player in that matchup. |
| `mean_steps` | Mean rollout length. |
| `attacker_goal_rate` | Fraction of eval rollouts ending in attacker goal. |
| `tagged_rate` | Fraction of eval rollouts ending in defender tag. |
| `timeout_rate` | Fraction of eval rollouts ending by timeout. |

Each matchup also logs one uncertainty metric under `eval_extra`:

| Metric | Meaning |
|---|---|
| `eval_extra/az_p1_vs_heuristic/stderr_reward` | Standard error of player 1 AlphaZero reward. |
| `eval_extra/heuristic_vs_az_p2/stderr_reward` | Standard error of player 2 AlphaZero reward. |

The eval callbacks intentionally do not log duplicated search epsilon values or
constant eval `max_steps`.

## Dubin Run Config

The W&B run config records static hyperparameters and run identity. Current
Dubin scripts log the following common config keys:

| Config key | Meaning |
|---|---|
| `experiment` | Experiment directory/name, for example `dubin-2026-06-15`. |
| `search/name` | Search variant name, for example `smoos` or `regret_matching`. |
| `search/type` | Search type, for example `SMOOSSearch` or `MCTSSearch`. |
| `game` | Game identifier, currently `DubinMG`. |
| `sim_depth` | Self-play simulation depth. |
| `max_steps` | Solver target sample budget. |
| `num_steps` | Self-play samples requested per solver update. |
| `update_epochs` | Training epochs per update. |
| `num_batches` | Training minibatches per epoch/update. |
| `width` | Neural network hidden width used by the experiment script. |
| `lr` | Learning rate. |
| `lr_decay` | Per-update learning-rate decay factor. |
| `lr_min` | Lower bound applied to the decayed learning rate. |
| `lr_max` | Upper bound applied to the decayed learning rate. |
| `ema` | Whether callback/final oracle uses EMA weights. |
| `ema_decay` | EMA decay factor. |
| `gae_lambda` | GAE lambda for SM-OOS target construction. |
| `critic_type` | Critic target/output type selected by the experiment script. |
| `epsilon_initial` | Initial exploration epsilon from the schedule. |
| `epsilon_decay` | Epsilon decay factor. |
| `eval_runs` | Number of heuristic eval rollouts per matchup. |
| `eval_every` | Solver iteration interval for heuristic eval logging. |

SM-OOS config keys:

| Config key | Meaning |
|---|---|
| `search/oos_iterations` | Number of OOS iterations per search. |
| `search/max_depth` | Search depth limit. |
| `search/transfer_weight` | Transfer-weight parameter for SM-OOS state updates. |
| `search/transfer_payoff_bound` | Payoff bound Δ used to project transferred regrets onto the theorem's weight condition (`Inf` disables). |
| `search/tau` | Initial transfer temperature/state. |
| `oracle/value_weight` | Value-loss weight. |
| `oracle/regret_weight` | Regret-loss weight. |
| `oracle/strategy_weight` | Strategy-loss weight. |

Regret-matching / MCTS config keys:

| Config key | Meaning |
|---|---|
| `search/tree_queries` | MCTS tree-query budget. |
| `search/max_depth` | Search depth limit. |
| `search/max_time` | MCTS wall-clock time limit. |
| `search/backup` | MCTS backup style used by `RegretMatchingSearch`. |
| `search/value_target` | Value target source, such as `search` or `rollout`. |
| `oracle/value_weight` | Value-loss weight. |
| `oracle/policy_weight` | Policy-loss weight. |

The older `experiments/dubin/dubin-2026-06-15/train.jl` script logs the SM-OOS
style config but does not include `search/name` or `search/type`.

## Practical Notes

- W&B history keys are metric names like `oracle_quality/value_pred_mse`; config
  keys are separate W&B run config entries.
- For run comparison, prefer mode-compatible metrics. For example, compare
  `training_health/mean_policy_loss` only for actor-critic runs and
  `training_health/mean_regret_loss` only for fitted-regret runs.
- `training_health/mean_loss` is mode-dependent because each training mode
  combines different component losses. Component losses and eval rewards are
  usually more interpretable.
- `oracle_quality/value_explained_variance` can be negative when the value
  function is worse than predicting a constant mean.
