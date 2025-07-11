@kwdef struct MCTSParams{T, Oracle, MS}
    tree_queries    :: Int      = 10
    c               :: Float64  = 1.0
    max_depth       :: Int      = 100
    temperature     :: T        = t -> 1.0 * (0.95 ^ (t-1))
    max_time        :: Float64  = Inf
    matrix_solver   :: MS       = RegretSolver(20)
    oracle          :: Oracle
end

uniform(n::Int) = fill(inv(n), n)

function search_info(params::MCTSParams, game::MG, s; temperature=1.0)
    (;matrix_solver) = params
    tree = Tree(game, s)
    x,y,v = if isterminal(game, s)
        n1, n2 = length.(actions(game))
        uniform(n1), uniform(n2), 0.0
    elseif iszero(params.max_depth) || iszero(params.tree_queries) || iszero(params.max_time)
        solve(matrix_solver, oracle_matrix_game(game, params.oracle, s))
    else
        (;tree_queries, c) = params
        γ = discount(game)
        for i ∈ 1:tree_queries
            simulate(params, tree, game, 1; temperature)
        end
        solve(matrix_solver, node_matrix_game(tree, c, 1, γ))
    end
    return (x,y,v), (;tree)
end

function search(params::MCTSParams, game::MG, s; temperature=1.0)
    return first(search_info(params, game, s; temperature))
end

function simulate(params, tree, game, s_idx; temperature=1.0)
    (; c, oracle, matrix_solver) = params
    γ = discount(game)
    s = tree.s[s_idx]
    ns = tree.n_s[s_idx]

    if isterminal(game, s)
        return 0.0
    elseif is_leaf(tree, s_idx)
        expand_s!(tree, s_idx, game, oracle)
        x,y,t = solve(matrix_solver, node_matrix_game(tree, c, s_idx, γ))
        return t
    else
        # choose best action for exploration
        a = explore_action(matrix_solver, tree, c, s_idx, γ; temperature)
        sp_idx = tree.s_children[s_idx][a]
        vp_sample = simulate(params, tree, game, sp_idx; temperature)


        # update node stats
        v̂ = tree.v[s_idx][a]
        nsa = tree.n_sa[s_idx][a]
        # tree.v[s_idx][a]     = v_sample # (v_sample - v̂) / (ns + 1)
        tree.v[s_idx][a]     = v̂ + (vp_sample - v̂) / (nsa + 1)
        tree.n_s[s_idx]     += 1
        tree.n_sa[s_idx][a] += 1

        # solve game with updated statistics
        # x,y,t = solve(matrix_solver, node_matrix_game(tree, c, s_idx, γ))
        return tree.r[s_idx][a] + γ * vp_sample 
    end
end

function ucb_exploration(tree::Tree, c::Float64, s_idx::Int)
    n_sa = tree.n_sa[s_idx]
    n_s = tree.n_s[s_idx]
    return c .* sqrt.(log(max(1,n_s)) ./ max.(1, n_sa))
end

function pucb_exploration(tree::Tree, c::Float64, s_idx::Int; temperature=1.0)
    Ē = ucb_exploration(tree, c, s_idx)
    σ1, σ2 = softmax.(getindex.(tree.prior, s_idx) ./ temperature)
    return (σ1 * σ2') .* Ē
end

function ucb_matrix_games(tree::Tree, c::Float64, s_idx::Int, γ::Float64; temperature=1.0)
    V = node_matrix_game(tree, c, s_idx, γ)
    Ē = pucb_exploration(tree, c, s_idx; temperature)
    return V .+ Ē, -V .+ Ē
end

function node_matrix_game(tree::Tree, c, s_idx, γ)
    r = tree.r[s_idx]
    v = tree.v[s_idx]
    return r .+ γ .* v
end

function explore_action(matrix_solver, tree::Tree, c::Float64, s_idx::Int, γ::Float64; temperature=1.0)
    x,y,t = solve(matrix_solver, ucb_matrix_games(tree, c, s_idx, γ; temperature)...)
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
        x,y,gv = search(params, game, s; temperature)
        x = softmax(x ./ temperature)
        y = softmax(y ./ temperature)
        
        a_idxs = Tuple(action_idx_from_probs(x,y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
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
        vp = only(value(params.oracle, MarkovGames.convert_s(Vector{Float32}, s, game)))
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

zs_reward_scalar(x::Number) = x
zs_reward_scalar(x::Union{Tuple, AbstractArray}) = first(x)
