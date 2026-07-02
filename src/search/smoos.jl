struct SMOOSTree{S}
    s           :: Vector{S}
    children    :: Vector{Dict{Tuple{Int,Int}, Int}}
    regret      :: NTuple{2, Vector{Vector{Float64}}}
    strategy    :: NTuple{2, Vector{Vector{Float64}}}
end

function SMOOSTree(game::MG, s=rand(initialstate(game)))
    return SMOOSTree(
        [s],
        [Dict{Tuple{Int,Int}, Int}()],
        (Vector{Float64}[Float64[]], Vector{Float64}[Float64[]]),
        (Vector{Float64}[Float64[]], Vector{Float64}[Float64[]]),
    )
end

Tree(::SMOOSSearch, game::MG, s=rand(initialstate(game))) = SMOOSTree(game, s)
Tree(game::MG, s=rand(initialstate(game))) = SMOOSTree(game, s)

# Project transferred regrets onto the feasible set of the regret-transfer
# theorem: the weight condition Φ(wR̂ᶜ) ≤ wT₁|A|Δ² requires the positive-part
# norm of the initialization to be at most sqrt(τ|A|)Δ (with wT₁ = τ).
function project_transfer_regret!(q::Vector{Float64}, τ::Float64, nA::Int, Δ::Float64)
    isfinite(Δ) || return q
    bound = sqrt(τ * nA) * Δ
    potential = sqrt(sum(x -> max(x, 0.0)^2, q))
    potential > bound && (q .*= bound / potential)
    return q
end

# The exact accumulator initialization injected at node expansion. Kept as a
# separate function so root_targets can subtract it back out of the training
# targets: the network must never be trained on its own prior.
function transfer_prior(params::SMOOSSearch, game::MG, s)
    A1, A2 = actions(game)
    r̂ = state_regret(params.oracle, game, s)
    ŝ = state_strategy(params.oracle, game, s)
    # params.τ is already discounted: τ = wM, so w*sqrt(M) = sqrt(w*τ).
    regret_mass = sqrt(params.transfer_weight * params.τ)
    Δ = params.transfer_payoff_bound
    r1 = project_transfer_regret!(regret_mass .* Float64.(r̂[1]), params.τ, length(A1), Δ)
    r2 = project_transfer_regret!(regret_mass .* Float64.(r̂[2]), params.τ, length(A2), Δ)
    s1 = params.τ .* normalized_or_uniform(Float64.(ŝ[1]))
    s2 = params.τ .* normalized_or_uniform(Float64.(ŝ[2]))
    return (r1, r2), (s1, s2)
end

function expand_node!(tree::SMOOSTree, h::Int, game::MG, params::SMOOSSearch)
    isempty(tree.regret[1][h]) || return nothing
    A1, A2 = actions(game)
    (r1, r2), (s1, s2) = transfer_prior(params, game, tree.s[h])
    tree.regret[1][h] = r1
    tree.regret[2][h] = r2
    tree.strategy[1][h] = s1
    tree.strategy[2][h] = s2
    @assert length(tree.regret[1][h]) == length(A1)
    @assert length(tree.regret[2][h]) == length(A2)
    return nothing
end

function child_index!(tree::SMOOSTree, h::Int, a::CartesianIndex{2}, sp)
    key = Tuple(a)
    return get!(tree.children[h], key) do
        push!(tree.s, sp)
        push!(tree.children, Dict{Tuple{Int,Int}, Int}())
        push!(tree.regret[1], Float64[])
        push!(tree.regret[2], Float64[])
        push!(tree.strategy[1], Float64[])
        push!(tree.strategy[2], Float64[])
        return length(tree.s)
    end
end

function root_targets(params::SMOOSSearch, tree::SMOOSTree, game::MG, h::Int=1)
    expand_node!(tree, h, game, params)
    # Targets are computed from fresh search evidence only: the prior mass
    # injected at expansion is subtracted back out so the oracle is never
    # trained on its own predictions (self-distillation echo chamber).
    (r1_init, r2_init), (s1_init, s2_init) = transfer_prior(params, game, tree.s[h])
    regret_denom = sqrt(max(Float64(params.oos_iterations), 1.0))
    yr = (
        (Float64.(tree.regret[1][h]) .- r1_init) ./ regret_denom,
        (Float64.(tree.regret[2][h]) .- r2_init) ./ regret_denom,
    )
    ys = (
        normalized_or_uniform(max.(Float64.(tree.strategy[1][h]) .- s1_init, 0.0)),
        normalized_or_uniform(max.(Float64.(tree.strategy[2][h]) .- s2_init, 0.0)),
    )
    A1, A2 = actions(game)
    length(yr[1]) == length(A1) || error("player 1 regret target has wrong action dimension")
    length(yr[2]) == length(A2) || error("player 2 regret target has wrong action dimension")
    return yr, ys
end

function fitted_smoos_info(params::SMOOSSearch, game::MG, s; ϵ=0.30)
    tree = Tree(params, game, s)
    if !isterminal(game, s)
        for _ ∈ 1:params.oos_iterations
            smoos_trajectory!(params, tree, game, 1, 0, 1.0, 1.0, 1.0, 1.0; ϵ)
        end
    end
    yr, ys = root_targets(params, tree, game, 1)
    return (yr, ys), (; tree)
end

fitted_smoos(params::SMOOSSearch, game::MG, s; ϵ=0.30) =
    first(fitted_smoos_info(params, game, s; ϵ))

function smoos_trajectory!(
        params::SMOOSSearch,
        tree::SMOOSTree,
        game::MG,
        h::Int,
        depth::Int,
        x1::Float64,
        x2::Float64,
        q1::Float64,
        q2::Float64;
        ϵ=0.30
    )
    s = tree.s[h]
    if isterminal(game, s)
        return 1.0, 1.0, 1.0, 1.0, 0.0
    elseif depth ≥ params.max_depth
        return 1.0, 1.0, 1.0, 1.0, oracle_state_value(params.oracle, game, s)
    end

    expand_node!(tree, h, game, params)
    σ1 = regret_matching_policy(tree.regret[1][h])
    σ2 = regret_matching_policy(tree.regret[2][h])
    σ1_sample = eps_exploration(σ1, ϵ)
    σ2_sample = eps_exploration(σ2, ϵ)

    a = action_idx_from_probs(σ1_sample, σ2_sample)
    i, j = Tuple(a)
    A1, A2 = actions(game)
    sp, r = @gen(:sp, :r)(game, s, (A1[i], A2[j]))
    r = zs_reward_scalar(r)

    hp = child_index!(tree, h, a, sp)
    tail_x1, tail_x2, tail_q1, tail_q2, tail_value = smoos_trajectory!(
        params, tree, game, hp, depth + 1,
        x1 * σ1[i], x2 * σ2[j],
        q1 * σ1_sample[i], q2 * σ2_sample[j];
        ϵ
    )

    value = Float64(r) + discount(game) * tail_value
    sample_reach = max(
        σ1_sample[i] * σ2_sample[j] * tail_q1 * tail_q2,
        eps(Float64),
    )
    w1 = value * (σ2[j] * tail_x2) * tail_x1 / sample_reach
    w2 = -value * (σ1[i] * tail_x1) * tail_x2 / sample_reach

    for b ∈ eachindex(σ1)
        tree.regret[1][h][b] += ((b == i) - σ1[b]) * w1
    end
    for b ∈ eachindex(σ2)
        tree.regret[2][h][b] += ((b == j) - σ2[b]) * w2
    end
    tree.strategy[1][h] .+= σ1
    tree.strategy[2][h] .+= σ2
    return tail_x1 * σ1[i], tail_x2 * σ2[j], tail_q1 * σ1_sample[i], tail_q2 * σ2_sample[j], value
end

function lambda_gae_targets(rewards, values, bootstrap, γ, λ)
    targets = Vector{Float64}(undef, length(rewards))
    adv = 0.0
    vnext = Float64(bootstrap)
    for t ∈ reverse(eachindex(rewards))
        δ = Float64(rewards[t]) + γ * vnext - Float64(values[t])
        adv = δ + γ * λ * adv
        targets[t] = adv + Float64(values[t])
        vnext = Float64(values[t])
    end
    return targets
end

function smoos_sim(search::SMOOSSearch, game::MG, s; progress=false, ϵ=0.30, sim_depth::Int=search.max_depth, gae_lambda=0.95)
    sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    rewards = Float64[]
    values = Float64[]
    search_time_hist = Float64[]
    s_hist = Vector{Float32}[]
    regret_hist = (Vector{Float64}[], Vector{Float64}[])
    strategy_hist = (Vector{Float64}[], Vector{Float64}[])
    p = Progress(sim_depth, enabled=progress)

    while (t <= sim_depth) && !isterminal(game, s)
        search_start = time()
        (yr, ys), _info = fitted_smoos_info(search, game, s; ϵ)
        search_time = time() - search_start

        # Mix in the same ϵ used for in-tree exploration so executed self-play
        # actions never collapse to a pure argmax; training targets (yr, ys)
        # are left unperturbed.
        x = eps_exploration(normalized_or_uniform(ys[1]), ϵ)
        y = eps_exploration(normalized_or_uniform(ys[2]), ϵ)
        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(values, oracle_state_value(search.oracle, game, s))
        push!(rewards, Float64(r))
        push!(regret_hist[1], Float64.(yr[1]))
        push!(regret_hist[2], Float64.(yr[2]))
        push!(strategy_hist[1], Float64.(ys[1]))
        push!(strategy_hist[2], Float64.(ys[2]))
        t += 1
        s = sp
        next!(p)
    end
    bootstrap = isterminal(game, s) ? 0.0 : oracle_state_value(search.oracle, game, s)
    v_hist = lambda_gae_targets(rewards, values, bootstrap, γ, gae_lambda)
    finish!(p)
    return (;
        s = s_hist,
        r = rewards,
        v = v_hist,
        search_time = search_time_hist,
        regret = regret_hist,
        strategy = strategy_hist,
    )
end
