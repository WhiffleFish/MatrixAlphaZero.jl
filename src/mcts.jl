@kwdef struct MCTSParams{Oracle}
    tree_queries    :: Int      = 10
    c               :: Float64  = 1.0
    oracle          :: Oracle
end

function search(params::MCTSParams, game::MG, s)
    (;tree_queries, c) = params
    γ = discount(game)
    tree = Tree(game, s)
    for i ∈ 1:tree_queries
        simulate(params, tree, game, 1)
    end
    return solve(ucb_matrix_game(tree, c, 1, γ))
end

function simulate(params, tree, game, s_idx)
    (; c, oracle) = params
    γ = discount(game)
    s = tree.s[s_idx]
    ns = tree.n_s[s_idx]

    if isterminal(game, s)
        return 0.0
    elseif is_leaf(tree, s_idx)
        expand_s!(tree, s_idx, game, oracle)
        x,y,t = solve(ucb_matrix_game(tree, c, s_idx, γ))
        return t
    else
        # choose best action for exploration
        a = explore_action(tree, c, s_idx, γ)
        sp_idx = tree.s_children[s_idx][a]
        v_sample = simulate(params, tree, game, sp_idx)

        # update node stats
        v̂ = tree.v[s_idx][a]
        tree.v[s_idx][a]    += (v_sample - v̂) / (ns + 1)
        tree.n_s[s_idx]     += 1
        tree.n_sa[s_idx][a] += 1

        # solve game with updated statistics
        x,y,t = solve(ucb_matrix_game(tree, c, s_idx, γ))
        return t
    end
end

function ucb_matrix_game(tree::Tree, c, s_idx::Int, γ)
    r = tree.r[s_idx]
    v = tree.v[s_idx]
    n_sa = tree.n_sa[s_idx]
    n_s = tree.n_s[s_idx]
    return r .+ γ .* v .+ c .* sqrt.(log(max(1,n_s)) ./ max.(1, n_sa))
end

function explore_action(tree, c, s_idx, γ)
    x,y,t = solve(ucb_matrix_game(tree, c, s_idx, γ))
    return CartesianIndex(
        rand(Categorical(x)), 
        rand(Categorical(y))
    )
end
