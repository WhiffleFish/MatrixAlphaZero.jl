# SM-OOS
- [ ] We don't need to store/update strategy sums at non-root nodes
- [ ] Need to update importance sampling term if doing eps-exploration for both agents OR alternate updates

# Wandb
- Remove
    - `training_health/minibatch_metrics` - just shows up as hard-to-read table
- `oos_epsilon` - currently in `eval` repeated for both heuristic evaluations. Only need it once
- re-order
    - For eval, want rewards front and center
- Check gradient clip rate

# Training
- Given high grad norms, seems like we're clipping pretty much every gradient

# Ideas
- Vary transfer_weight based on regret error
    - Start with small transfer_weight and gradually increase as we get closer to convergence
