function zero_query_search(oracle, game::MG, s)
    x, y = state_policy(oracle, game, s)
    return Float64.(x), Float64.(y), oracle_state_value(oracle, game, s)
end

function search_info(params::MCTSParams, game::MG, s; ϵ=0.30)
    tree = Tree(params, game, s)
    x, y, v = if isterminal(game, s)
        n1, n2 = length.(actions(game))
        uniform(n1), uniform(n2), 0.0
    elseif iszero(params.max_depth) || iszero(params.tree_queries) || iszero(params.max_time)
        zero_query_search(params.oracle, game, s)
    else
        for _ ∈ 1:params.tree_queries
            simulate(params, tree, game, 1; ϵ)
        end
        search_result(params, tree, game, 1; ϵ)
    end
    return (x, y, v), (; tree)
end

function search(params::MCTSParams, game::MG, s; ϵ=0.30)
    return first(search_info(params, game, s; ϵ))
end

function simulate(params::MCTSParams, tree::AbstractSearchTree, game::MG, s_idx; ϵ=0.30)
    return simulate(params.search_style, params, tree, game, s_idx, 0; ϵ)
end

function simulate(style::RegretMatchingSearch, params::MCTSParams, tree::RegretMatchingTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30)
    return simulate_regret_matching(style, params, tree, game, s_idx, depth; ϵ)
end

function simulate_regret_matching(style::RegretMatchingSearch, params::MCTSParams, tree::RegretMatchingTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30)
    s = tree.s[s_idx]
    if isterminal(game, s)
        return 0.0
    elseif depth >= params.max_depth
        leaf_value = oracle_state_value(params.oracle, game, s)
        add_return_sum!(tree, s_idx, leaf_value)
        tree.n_s[s_idx] += 1
        return leaf_value
    elseif is_leaf(tree, s_idx)
        expand_s!(tree, s_idx, game, params.oracle)
        leaf_value = oracle_state_value(params.oracle, game, s)
        add_return_sum!(tree, s_idx, leaf_value)
        tree.n_s[s_idx] += 1
        return leaf_value
    else
        γ = discount(game)
        π1, π2 = selection_policy(style, tree, s_idx; ϵ)
        a = action_idx_from_probs(π1, π2)
        sp_idx = tree.s_children[s_idx][a]
        vp = simulate(style, params, tree, game, sp_idx, depth + 1; ϵ)
        total = tree.r[s_idx][a] + γ * vp

        v̂ = tree.v[s_idx][a]
        nsa = tree.n_sa[s_idx][a]
        tree.v[s_idx][a] = v̂ + (vp - v̂) / (nsa + 1)
        tree.n_s[s_idx] += 1
        tree.n_sa[s_idx][a] += 1
        add_return_sum!(tree, s_idx, total)

        update_node!(style, tree, s_idx, a, total, π1, π2, γ)
        return backup_value(style, tree, s_idx, total)
    end
end

function search_result(params::MCTSParams, tree::AbstractSearchTree, game::MG, s_idx::Int; ϵ=0.30)
    x, y = tree_policy(params, tree, game, s_idx; ϵ)
    v = node_value(params, tree, game, s_idx, x, y)
    return x, y, v
end

function tree_policy(params::MCTSParams, tree::AbstractSearchTree, game::MG, s_idx::Int; ϵ=0.30)
    return tree_policy(params.search_style, params, tree, game, s_idx; ϵ)
end

function node_value(params::MCTSParams, tree::AbstractSearchTree, game::MG, s_idx::Int, x, y)
    return node_value(params.search_style, params, tree, game, s_idx, x, y)
end

function node_value(::RegretMatchingSearch, params::MCTSParams, tree::RegretMatchingTree, game::MG, s_idx::Int, x, y)
    if iszero(tree.n_s[s_idx])
        return oracle_state_value(params.oracle, game, tree.s[s_idx])
    end
    return node_return_sum(tree, s_idx) / tree.n_s[s_idx]
end

function backup_value(style::RegretMatchingSearch, tree::RegretMatchingTree, s_idx::Int, sample_value::Float64)
    return style.backup == :mean ? node_return_sum(tree, s_idx) / tree.n_s[s_idx] : sample_value
end

root_policy(style::AbstractSearchStyle, x, y, ϵ::Real) = (x, y)

function mcts_sim(params::MCTSParams, game::MG, s; progress=false, ϵ=0.30)
    d = params.max_depth
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    v = 0.0
    r_hist = Float64[]
    v_hist = Float64[]
    s_hist = Vector{Float32}[]
    policy_hist = (
        Vector{Float64}[],
        Vector{Float64}[],
    )
    use_search_targets = params.value_target == :search
    if !use_search_targets && params.value_target != :rollout
        throw(ArgumentError("Unsupported value_target=$(params.value_target). Use :search or :rollout."))
    end
    p = Progress(d, enabled=progress)

    while (t < d) && !isterminal(game, s)
        x, y, gv = search(params, game, s; ϵ)
        x, y = root_policy(params.search_style, x, y, ϵ)

        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        v += r * γ^(t - 1)
        push!(r_hist, r)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(v_hist, use_search_targets ? gv : 0.0)
        push!(policy_hist[1], x)
        push!(policy_hist[2], y)
        if !use_search_targets
            for _t ∈ 1:t
                v_hist[_t] += r * γ^(t - _t)
            end
        end
        t += 1
        s = sp
        next!(p)
    end
    if !use_search_targets && !isterminal(game, s)
        vp = only(value(params.oracle, MarkovGames.convert_s(Vector{Float32}, s, game)))
        for _t ∈ 1:(d - 1)
            v_hist[_t] += vp * γ^(t - _t)
        end
    end
    finish!(p)
    return (;
        s = s_hist,
        r = r_hist,
        v = v_hist,
        policy = policy_hist,
    )
end
