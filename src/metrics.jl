const NaN32 = Float32(NaN)

kl_divergence(p, q) = sum(eachindex(p, q)) do i
    p_i = p[i]
    iszero(p_i) ? zero(p_i) : p_i * log(p_i / max(q[i], eps(eltype(p))))
end

strategy_entropy(p::AbstractVector{<:Real}) = -sum(x -> iszero(x) ? zero(x) : x * log(x), p)
strategy_entropy(strategies::AbstractVector{<:AbstractVector}) = mean(strategy_entropy, strategies)
mean_l2_norm(x::AbstractMatrix) = mean(sqrt(sum(abs2, col)) for col ∈ eachcol(x))

function explained_variance(pred, target)
    μ = sum(target) / length(target)
    target_var = sum(abs2, target .- μ) / length(target)
    target_var > eps(Float32) || return NaN32
    residual = target .- pred
    residual_μ = sum(residual) / length(residual)
    residual_var = sum(abs2, residual .- residual_μ) / length(residual)
    return Float32(1 - residual_var / target_var)
end

function selfplay_metrics(hists)
    ep_lengths = map(h -> length(h.v), hists)
    rewards = mapreduce(h -> h.r, vcat, hists)
    search_times = mapreduce(h -> h.search_time, vcat, hists)
    return (;
        mean_ep_length    = Float32(mean(ep_lengths)),
        mean_reward       = Float32(mean(rewards)),
        reward_std        = Float32(std(rewards)),
        mean_search_time  = Float32(mean(search_times)),
        total_search_time = Float32(sum(search_times)),
        search_count      = length(search_times),
    )
end

function training_metrics(train_stats)
    base = (;
        mean_loss          = Float32(mean(train_stats[:losses])),
        mean_value_loss    = Float32(mean(train_stats[:value_losses])),
        mean_grad_norm     = Float32(mean(train_stats[:grad_norms])),
        max_grad_norm      = Float32(maximum(train_stats[:grad_norms])),
    )
    if haskey(train_stats, :regret_losses)
        return merge(base, (;
            mean_regret_loss   = Float32(mean(train_stats[:regret_losses])),
            mean_strategy_loss = Float32(mean(train_stats[:strategy_losses])),
        ))
    else
        return merge(base, (;
            mean_policy_loss = Float32(mean(train_stats[:policy_losses])),
        ))
    end
end

function training_minibatch_metrics(train_stats)
    base = (;
        minibatch     = collect(1:length(train_stats[:losses])),
        loss          = Float32.(train_stats[:losses]),
        value_loss    = Float32.(train_stats[:value_losses]),
        grad_norm     = Float32.(train_stats[:grad_norms]),
    )
    if haskey(train_stats, :regret_losses)
        return merge(base, (;
            regret_loss   = Float32.(train_stats[:regret_losses]),
            strategy_loss = Float32.(train_stats[:strategy_losses]),
        ))
    else
        return merge(base, (;
            policy_loss = Float32.(train_stats[:policy_losses]),
        ))
    end
end

function batch_metrics(batch::NamedTuple)
    return (;
        batch_size = length(batch.v),
    )
end

function oracle_metrics(oracle, prev_oracle, batch::NamedTuple; n_samples::Int=128)
    batch_size = length(batch.v)
    iszero(batch_size) && return empty_oracle_metrics(batch)
    n    = min(n_samples, batch_size)
    idxs = rand(1:batch_size, n)
    X    = reduce(hcat, batch.s[idxs])

    target_policy = hasproperty(batch, :strategy) ? batch.strategy : batch.policy

    h1 = Float32(strategy_entropy(target_policy[1]))
    h2 = Float32(strategy_entropy(target_policy[2]))

    s_cur  = policy(oracle, X)
    s_prev = policy(prev_oracle, X)
    kl_p1 = Float32(mean(kl_divergence(s_cur[1][:, i], s_prev[1][:, i]) for i ∈ 1:n))
    kl_p2 = Float32(mean(kl_divergence(s_cur[2][:, i], s_prev[2][:, i]) for i ∈ 1:n))

    s_t1 = Float32.(reduce(hcat, target_policy[1][idxs]))
    s_t2 = Float32.(reduce(hcat, target_policy[2][idxs]))
    foreach(col -> normalize_or_uniform!(col), eachcol(s_t1))
    foreach(col -> normalize_or_uniform!(col), eachcol(s_t2))
    skl_p1 = Float32(mean(kl_divergence(s_cur[1][:, i], s_t1[:, i]) for i ∈ 1:n))
    skl_p2 = Float32(mean(kl_divergence(s_cur[2][:, i], s_t2[:, i]) for i ∈ 1:n))

    v_target = batch.v[idxs]
    v_pred = vec(value(oracle, X))
    v_mse  = Float32(mean(abs2, v_pred .- v_target))

    policy_metrics = (;
        policy_entropy_p1   = h1,
        policy_entropy_p2   = h2,
        policy_kl_p1        = kl_p1,
        policy_kl_p2        = kl_p2,
        target_policy_kl_p1 = skl_p1,
        target_policy_kl_p2 = skl_p2,
        value_pred_mse      = v_mse,
        value_explained_variance = explained_variance(v_pred, v_target),
    )
    hasproperty(batch, :regret) || return policy_metrics

    r_cur = regret(oracle, X)
    r_t1 = Float32.(reduce(hcat, batch.regret[1][idxs]))
    r_t2 = Float32.(reduce(hcat, batch.regret[2][idxs]))
    return merge(policy_metrics, (;
        target_regret_l2 = Float32(0.5 * (mean_l2_norm(r_t1) + mean_l2_norm(r_t2))),
        regret_pred_mse  = Float32(0.5 * (mean(abs2, r_cur[1] .- r_t1) + mean(abs2, r_cur[2] .- r_t2))),
    ))
end

function empty_oracle_metrics(batch::NamedTuple)
    base = (;
        policy_entropy_p1   = NaN32,
        policy_entropy_p2   = NaN32,
        policy_kl_p1        = NaN32,
        policy_kl_p2        = NaN32,
        target_policy_kl_p1 = NaN32,
        target_policy_kl_p2 = NaN32,
        value_pred_mse      = NaN32,
        value_explained_variance = NaN32,
    )
    hasproperty(batch, :regret) || return base
    return merge(base, (; target_regret_l2=NaN32, regret_pred_mse=NaN32))
end
