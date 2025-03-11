@proto @kwdef struct MCTSParams{Oracle}
    tree_queries    :: Int      = 10
    c               :: Float64  = 1.0
    max_depth       :: Int      = 100
    oracle          :: Oracle
end

function search(params::MCTSParams, game::MG, s)
    (;tree_queries, c) = params
    γ = discount(game)
    tree = Tree(game, s)
    for i ∈ 1:tree_queries
        simulate(params, tree, game, 1)
    end
    return solve(ucb_matrix_games(tree, c, 1, γ)...)
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
        x,y,t = solve(node_matrix_game(tree, c, s_idx, γ))
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
        x,y,t = solve(node_matrix_game(tree, c, s_idx, γ))
        return t
    end
end

function ucb_exploration(tree::Tree, c::Float64, s_idx::Int)
    n_sa = tree.n_sa[s_idx]
    n_s = tree.n_s[s_idx]
    return c .* sqrt.(log(max(1,n_s)) ./ max.(1, n_sa))
end

function ucb_matrix_games(tree::Tree, c::Float64, s_idx::Int, γ::Float64)
    V = node_matrix_game(tree, c, s_idx, γ)
    Ē = ucb_exploration(tree, c, s_idx)
    return V .+ Ē, -V .+ Ē
end

function node_matrix_game(tree::Tree, c, s_idx, γ)
    r = tree.r[s_idx]
    v = tree.v[s_idx]
    return r .+ γ .* v
end

function explore_action(tree, c, s_idx, γ)
    x,y,t = solve(ucb_matrix_games(tree, c, s_idx, γ)...)
    return action_idx_from_probs(x,y)
end

function action_idx_from_probs(x,y)
    return CartesianIndex(
        rand(Categorical(x)), 
        rand(Categorical(y))
    )
end

# TODO: change name
function mcts_sim(params::MCTSParams, game::MG, s; progress=false, temperature=1.0)
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
        Vector{Float64}[]
    )
    p = Progress(d, enabled=progress)

    while (t < d) && !isterminal(game, s)
        x,y,gv = search(params, game, s)
        x = softmax(x ./ temperature)
        y = softmax(y ./ temperature)
        
        a_idxs = Tuple(action_idx_from_probs(x,y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        v += r * γ^(t-1)
        push!(r_hist, r)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(v_hist, 0.0)
        push!(policy_hist[1], x)
        push!(policy_hist[2], y)
        for _t ∈ 1:t
            v_hist[_t] += r * γ^(t - _t)
        end
        t += 1
        s = sp
        next!(p)
    end
    if !isterminal(game, s)
        vp = only(params.oracle(MarkovGames.convert_s(Vector{Float32}, s, game)))
        for _t ∈ 1:(d-1)
            v_hist[_t] += vp * γ^(t - _t)
        end
    end
    finish!(p)
    return (;
        s       = s_hist,
        r       = r_hist,
        v       = v_hist,
        policy  = policy_hist
    )
end
