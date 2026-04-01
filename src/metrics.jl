kl_divergence(p, q) = sum(eachindex(p, q)) do i
    pi = p[i]
    iszero(pi) ? zero(pi) : pi * log(pi / max(q[i], eps(eltype(p))))
end

policy_entropy(p::AbstractVector{<:Real}) = -sum(x -> iszero(x) ? zero(x) : x * log(x), p)
policy_entropy(policies::AbstractVector{<:AbstractVector}) = mean(policy_entropy, policies)

function selfplay_metrics(hists)
    ep_lengths = map(h -> length(h.v), hists)
    rewards = mapreduce(h -> h.r, vcat, hists)
    return (;
        mean_ep_length = Float32(mean(ep_lengths)),
        mean_reward    = Float32(mean(rewards)),
        reward_std     = Float32(std(rewards)),
    )
end

function training_metrics(train_info)
    lv = Float32(mean(train_info[:value_losses]))
    lp = Float32(mean(train_info[:policy_losses]))
    return (;
        mean_loss          = Float32(mean(train_info[:losses])),
        mean_value_loss    = lv,
        mean_policy_loss   = lp,
        value_policy_ratio = lv / (lp + eps(Float32)),
        mean_grad_norm     = Float32(mean(train_info[:grad_norms])),
        max_grad_norm      = Float32(maximum(train_info[:grad_norms])),
    )
end

function buffer_metrics(buf::Buffer, samples_added::Int, capacity::Int)
    return (;
        buffer_size     = length(buf),
        buffer_turnover = Float32(samples_added) / capacity,
    )
end

# Computes oracle-based metrics using a random sample from the buffer.
# Requires oracle to support policy(oracle, X) and value(oracle, X) with batched input.
function oracle_metrics(oracle, prev_oracle, buf::Buffer; n_samples::Int=128)
    iszero(length(buf)) && return (;
        policy_entropy_p1   = NaN32, policy_entropy_p2   = NaN32,
        policy_kl_p1        = NaN32, policy_kl_p2        = NaN32,
        search_oracle_kl_p1 = NaN32, search_oracle_kl_p2 = NaN32,
        value_pred_mse      = NaN32,
    )
    n    = min(n_samples, length(buf))
    idxs = rand(1:length(buf), n)
    X    = reduce(hcat, buf.s[idxs])

    # Policy entropy from stored search-backed policy targets
    h1 = Float32(policy_entropy(buf.p[1]))
    h2 = Float32(policy_entropy(buf.p[2]))

    # Policy change: KL between current and previous EMA oracle
    p_cur  = policy(oracle, X)
    p_prev = policy(prev_oracle, X)
    kl_p1 = Float32(mean(kl_divergence(p_cur[1][:, i], p_prev[1][:, i]) for i ∈ 1:n))
    kl_p2 = Float32(mean(kl_divergence(p_cur[2][:, i], p_prev[2][:, i]) for i ∈ 1:n))

    # Search vs oracle: KL between current oracle policy and stored search targets
    p_s1   = reduce(hcat, buf.p[1][idxs])
    p_s2   = reduce(hcat, buf.p[2][idxs])
    skl_p1 = Float32(mean(kl_divergence(p_cur[1][:, i], p_s1[:, i]) for i ∈ 1:n))
    skl_p2 = Float32(mean(kl_divergence(p_cur[2][:, i], p_s2[:, i]) for i ∈ 1:n))

    # Value prediction error against stored value targets
    v_pred = vec(value(oracle, X))
    v_mse  = Float32(mean(abs2, v_pred .- buf.v[idxs]))

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
