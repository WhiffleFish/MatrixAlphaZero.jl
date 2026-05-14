kl_divergence(p, q) = sum(eachindex(p, q)) do i
    pi = p[i]
    iszero(pi) ? zero(pi) : pi * log(pi / max(q[i], eps(eltype(p))))
end

policy_entropy(p::AbstractVector{<:Real}) = -sum(x -> iszero(x) ? zero(x) : x * log(x), p)
policy_entropy(policies::AbstractVector{<:AbstractVector}) = mean(policy_entropy, policies)

"""
    selfplay_metrics(hists) -> NamedTuple

Aggregate statistics over the self-play trajectories collected in one training iteration.

| Field            | Description                                                  |
|:-----------------|:-------------------------------------------------------------|
| `mean_ep_length` | Mean number of steps per episode across all rollouts.        |
| `mean_reward`    | Mean per-step reward pooled across all steps and episodes.   |
| `reward_std`     | Standard deviation of per-step rewards (spread of outcomes). |
| `mean_search_time` | Mean wall-clock seconds spent per tree search.             |
"""
function selfplay_metrics(hists)
    ep_lengths = map(h -> length(h.v), hists)
    rewards = mapreduce(h -> h.r, vcat, hists)
    search_times = mapreduce(h -> h.search_time, vcat, hists)
    return (;
        mean_ep_length   = Float32(mean(ep_lengths)),
        mean_reward      = Float32(mean(rewards)),
        reward_std       = Float32(std(rewards)),
        mean_search_time = Float32(mean(search_times)),
        total_search_time = Float32(sum(search_times)),
        search_count     = length(search_times),
    )
end

"""
    training_metrics(train_stats) -> NamedTuple

Aggregate statistics over the gradient update steps within one training iteration.

| Field            | Description                                                                           |
|:-----------------|:--------------------------------------------------------------------------------------|
| `mean_loss`      | Mean total loss (value + policy + L2 regularisation) across all mini-batches.         |
| `mean_value_loss`| Mean value head loss (Huber or cross-entropy depending on critic type).               |
| `mean_policy_loss`| Mean policy head loss (cross-entropy against search-backed policy targets).          |
| `mean_grad_norm` | Mean ℓ₂ norm of the gradient vector **before** the optimiser clips it. A sustained   |
|                  | rise signals instability; zero means the network has stopped learning.                |
| `max_grad_norm`  | Maximum gradient norm seen across all mini-batches in the iteration. Useful for       |
|                  | detecting isolated spikes that the mean would smooth over.                            |
"""
function training_metrics(train_stats)
    return (;
        mean_loss         = Float32(mean(train_stats[:losses])),
        mean_value_loss   = Float32(mean(train_stats[:value_losses])),
        mean_policy_loss  = Float32(mean(train_stats[:policy_losses])),
        mean_grad_norm    = Float32(mean(train_stats[:grad_norms])),
        max_grad_norm     = Float32(maximum(train_stats[:grad_norms])),
    )
end

function training_minibatch_metrics(train_stats)
    return (;
        minibatch   = collect(1:length(train_stats[:losses])),
        loss        = Float32.(train_stats[:losses]),
        value_loss  = Float32.(train_stats[:value_losses]),
        policy_loss = Float32.(train_stats[:policy_losses]),
        grad_norm   = Float32.(train_stats[:grad_norms]),
    )
end

"""
    batch_metrics(batch) -> NamedTuple

Fresh training-batch size for the current iteration.

| Field             | Description                                                                      |
|:------------------|:---------------------------------------------------------------------------------|
| `batch_size`      | Number of current-iteration transitions used for training.                       |
"""
function batch_metrics(batch::NamedTuple)
    return (;
        batch_size = length(batch.v),
    )
end

"""
    oracle_metrics(oracle, prev_oracle, batch; n_samples=128) -> NamedTuple

Oracle quality metrics estimated from a random sample of `n_samples` transitions in the
current training batch. Requires the oracle to support batched `policy(oracle, X)` and
`value(oracle, X)` calls (satisfied by `ActorCritic`).

| Field                | Description                                                                    |
|:---------------------|:-------------------------------------------------------------------------------|
| `policy_entropy_p1/2`| Shannon entropy of the search-backed policy targets in the current batch,      |
|                      | averaged over current samples. High entropy = exploring broadly;               |
|                      | collapsing entropy = converging (or mode-collapsing) to a near-deterministic   |
|                      | policy.                                                                        |
| `policy_kl_p1/2`     | KL divergence D(oracle_cur ‖ oracle_prev) at the sampled states. Measures how  |
|                      | much the EMA oracle's policy distribution shifted this iteration. Large values |
|                      | relative to `policy_entropy` suggest the oracle is changing faster than the    |
|                      | data distribution.                                                             |
| `search_oracle_kl_p1/2` | KL divergence D(oracle_cur ‖ search_target) at the sampled states. Measures|
|                      | how far the oracle's current predictions are from the search-backed targets it |
|                      | was trained on. A rising value means the oracle is drifting away from what the |
|                      | search found, which can signal training instability.                           |
| `value_pred_mse`     | Mean squared error between the oracle's value predictions and the value targets |
|                      | in the current batch. Directly measures value-head fit on fresh data.          |
"""
function oracle_metrics(oracle, prev_oracle, batch::NamedTuple; n_samples::Int=128)
    batch_size = length(batch.v)
    iszero(batch_size) && return (;
        policy_entropy_p1   = NaN32, policy_entropy_p2   = NaN32,
        policy_kl_p1        = NaN32, policy_kl_p2        = NaN32,
        search_oracle_kl_p1 = NaN32, search_oracle_kl_p2 = NaN32,
        value_pred_mse      = NaN32,
    )
    n    = min(n_samples, batch_size)
    idxs = rand(1:batch_size, n)
    X    = reduce(hcat, batch.s[idxs])

    # Policy entropy from stored search-backed policy targets
    h1 = Float32(policy_entropy(batch.policy[1]))
    h2 = Float32(policy_entropy(batch.policy[2]))

    # Policy change: KL between current and previous EMA oracle
    p_cur  = policy(oracle, X)
    p_prev = policy(prev_oracle, X)
    kl_p1 = Float32(mean(kl_divergence(p_cur[1][:, i], p_prev[1][:, i]) for i ∈ 1:n))
    kl_p2 = Float32(mean(kl_divergence(p_cur[2][:, i], p_prev[2][:, i]) for i ∈ 1:n))

    # Search vs oracle: KL between current oracle policy and stored search targets
    p_s1   = reduce(hcat, batch.policy[1][idxs])
    p_s2   = reduce(hcat, batch.policy[2][idxs])
    skl_p1 = Float32(mean(kl_divergence(p_cur[1][:, i], p_s1[:, i]) for i ∈ 1:n))
    skl_p2 = Float32(mean(kl_divergence(p_cur[2][:, i], p_s2[:, i]) for i ∈ 1:n))

    # Value prediction error against stored value targets
    v_pred = vec(value(oracle, X))
    v_mse  = Float32(mean(abs2, v_pred .- batch.v[idxs]))

    return (;
        policy_entropy_p1   = h1,
        policy_entropy_p2   = h2,
        policy_kl_p1        = kl_p1,
        policy_kl_p2        = kl_p2,
        search_oracle_kl_p1 = skl_p1,
        search_oracle_kl_p2 = skl_p2,
        value_pred_mse      = v_mse,
    )
end
