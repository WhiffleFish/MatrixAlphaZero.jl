const NaN32 = Float32(NaN)

kl_divergence(p, q) = sum(eachindex(p, q)) do i
    pi = p[i]
    iszero(pi) ? zero(pi) : pi * log(pi / max(q[i], eps(eltype(p))))
end

strategy_entropy(p::AbstractVector{<:Real}) = -sum(x -> iszero(x) ? zero(x) : x * log(x), p)
strategy_entropy(strategies::AbstractVector{<:AbstractVector}) = mean(strategy_entropy, strategies)

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
    return (;
        mean_loss          = Float32(mean(train_stats[:losses])),
        mean_value_loss    = Float32(mean(train_stats[:value_losses])),
        mean_regret_loss   = Float32(mean(train_stats[:regret_losses])),
        mean_strategy_loss = Float32(mean(train_stats[:strategy_losses])),
        mean_grad_norm     = Float32(mean(train_stats[:grad_norms])),
        max_grad_norm      = Float32(maximum(train_stats[:grad_norms])),
    )
end

function training_minibatch_metrics(train_stats)
    return (;
        minibatch     = collect(1:length(train_stats[:losses])),
        loss          = Float32.(train_stats[:losses]),
        value_loss    = Float32.(train_stats[:value_losses]),
        regret_loss   = Float32.(train_stats[:regret_losses]),
        strategy_loss = Float32.(train_stats[:strategy_losses]),
        grad_norm     = Float32.(train_stats[:grad_norms]),
    )
end

function batch_metrics(batch::NamedTuple)
    return (;
        batch_size = length(batch.v),
    )
end

function oracle_metrics(oracle, prev_oracle, batch::NamedTuple; n_samples::Int=128)
    batch_size = length(batch.v)
    iszero(batch_size) && return (;
        strategy_entropy_p1   = NaN32, strategy_entropy_p2   = NaN32,
        strategy_kl_p1        = NaN32, strategy_kl_p2        = NaN32,
        target_strategy_kl_p1 = NaN32, target_strategy_kl_p2 = NaN32,
        regret_pred_mse       = NaN32,
        value_pred_mse        = NaN32,
    )
    n    = min(n_samples, batch_size)
    idxs = rand(1:batch_size, n)
    X    = reduce(hcat, batch.s[idxs])

    h1 = Float32(strategy_entropy(batch.strategy[1]))
    h2 = Float32(strategy_entropy(batch.strategy[2]))

    s_cur  = strategy(oracle, X)
    s_prev = strategy(prev_oracle, X)
    kl_p1 = Float32(mean(kl_divergence(s_cur[1][:, i], s_prev[1][:, i]) for i ∈ 1:n))
    kl_p2 = Float32(mean(kl_divergence(s_cur[2][:, i], s_prev[2][:, i]) for i ∈ 1:n))

    s_t1 = Float32.(reduce(hcat, batch.strategy[1][idxs]))
    s_t2 = Float32.(reduce(hcat, batch.strategy[2][idxs]))
    foreach(col -> normalize_or_uniform!(col), eachcol(s_t1))
    foreach(col -> normalize_or_uniform!(col), eachcol(s_t2))
    skl_p1 = Float32(mean(kl_divergence(s_cur[1][:, i], s_t1[:, i]) for i ∈ 1:n))
    skl_p2 = Float32(mean(kl_divergence(s_cur[2][:, i], s_t2[:, i]) for i ∈ 1:n))

    r_cur = regret(oracle, X)
    r_t1 = Float32.(reduce(hcat, batch.regret[1][idxs]))
    r_t2 = Float32.(reduce(hcat, batch.regret[2][idxs]))
    r_mse = Float32(0.5 * (mean(abs2, r_cur[1] .- r_t1) + mean(abs2, r_cur[2] .- r_t2)))

    v_pred = vec(value(oracle, X))
    v_mse  = Float32(mean(abs2, v_pred .- batch.v[idxs]))

    return (;
        strategy_entropy_p1   = h1,
        strategy_entropy_p2   = h2,
        strategy_kl_p1        = kl_p1,
        strategy_kl_p2        = kl_p2,
        target_strategy_kl_p1 = skl_p1,
        target_strategy_kl_p2 = skl_p2,
        regret_pred_mse       = r_mse,
        value_pred_mse        = v_mse,
    )
end
