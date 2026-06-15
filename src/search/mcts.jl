struct SearchTree{S}
    s           :: Vector{S}
    s_children  :: Vector{Matrix{Int}}
    n_sa        :: Vector{Matrix{Int}}
    n_s         :: Vector{Int}
    prior       :: NTuple{2, Vector{Vector{Float32}}}
    v           :: Vector{Matrix{Float64}}
    r           :: Vector{Matrix{Float64}}
    return_sum  :: Vector{Float64}
    regret      :: NTuple{2, Vector{Vector{Float64}}}
    policy_sum  :: NTuple{2, Vector{Vector{Float64}}}
end

const NO_CHILDREN = Matrix{Int}(undef, 0, 0)
const NO_PRIOR = Vector{Float32}(undef, 0)
const NO_FLOAT = Vector{Float64}(undef, 0)

function SearchTree(game::MG, s=rand(initialstate(game)))
    return SearchTree(
        [s],
        Matrix{Int}[NO_CHILDREN],
        Matrix{Int}[NO_CHILDREN],
        [0],
        ([NO_PRIOR], [NO_PRIOR]),
        [Matrix{Float64}(undef, 0, 0)],
        [Matrix{Float64}(undef, 0, 0)],
        [0.0],
        ([NO_FLOAT], [NO_FLOAT]),
        ([NO_FLOAT], [NO_FLOAT]),
    )
end

Tree(::MCTSSearch, game::MG, s=rand(initialstate(game))) = SearchTree(game, s)

is_leaf(tree::SearchTree, s_idx::Int) = isempty(tree.s_children[s_idx])

function reset_search_node!(tree::SearchTree, s_idx::Int, na1::Int, na2::Int)
    tree.return_sum[s_idx] = 0.0
    tree.regret[1][s_idx] = zeros(Float64, na1)
    tree.regret[2][s_idx] = zeros(Float64, na2)
    tree.policy_sum[1][s_idx] = zeros(Float64, na1)
    tree.policy_sum[2][s_idx] = zeros(Float64, na2)
    return nothing
end

function append_search_frontier!(tree::SearchTree, n_frontier::Int)
    append!(tree.return_sum, fill(0.0, n_frontier))
    foreach(tree.regret) do regret_i
        append!(regret_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.policy_sum) do policy_i
        append!(policy_i, fill(NO_FLOAT, n_frontier))
    end
    return nothing
end

function expand_s!(tree::SearchTree, s_idx::Int, game::MG, oracle)
    if is_leaf(tree, s_idx)
        _expand_s!(tree, s_idx, game, oracle)
    end
end

function _expand_s!(tree::SearchTree, s_idx::Int, game::MG, oracle)
    s = tree.s[s_idx]
    A1, A2 = actions(game)
    na1, na2 = length(A1), length(A2)
    s_children = zeros(Int, na1, na2)
    r = zeros(Float64, na1, na2)
    v = zeros(Float64, na1, na2)
    counter = length(tree.s) + 1
    frontier = statetype(game)[]
    nonterminal = trues(na1 * na2)

    flat_idx = 1
    for (j, a2) ∈ enumerate(A2), (i, a1) ∈ enumerate(A1)
        s_children[i, j] = counter
        sp, r_i = @gen(:sp, :r)(game, s, (a1, a2))
        push!(frontier, sp)
        r[i, j] = zs_reward_scalar(r_i)
        if isterminal(game, sp)
            nonterminal[flat_idx] = false
        end
        counter += 1
        flat_idx += 1
    end
    n_frontier = length(frontier)

    v̂ = batch_state_value(oracle, game, frontier)
    for i ∈ eachindex(v̂, nonterminal)
        v[i] = v̂[i] * nonterminal[i]
    end

    prior = state_policy(oracle, game, s)
    foreach(tree.prior, prior) do tree_prior, prior_i
        tree_prior[s_idx] = prior_i
    end

    tree.s_children[s_idx] = s_children
    tree.n_sa[s_idx] = zeros(Int, na1, na2)
    tree.n_s[s_idx] = 0
    tree.v[s_idx] = v
    tree.r[s_idx] = r
    reset_search_node!(tree, s_idx, na1, na2)

    append!(tree.s, frontier)
    append!(tree.s_children, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_sa, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_s, fill(0, n_frontier))
    append!(tree.v, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.r, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    foreach(tree.prior) do prior_i
        append!(prior_i, fill(NO_PRIOR, n_frontier))
    end
    append_search_frontier!(tree, n_frontier)
    return nothing
end

function oracle_policy(params::MCTSSearch, game::MG, tree::SearchTree, s_idx::Int)
    x, y = state_policy(params.oracle, game, tree.s[s_idx])
    return Float64.(x), Float64.(y)
end

function empirical_policy(tree::SearchTree, s_idx::Int)
    counts = tree.n_sa[s_idx]
    x = vec(sum(counts; dims=2))
    y = vec(sum(counts; dims=1))
    return normalize_or_uniform!(Float64.(x)), normalize_or_uniform!(Float64.(y))
end

function node_matrix_game(tree::SearchTree, s_idx::Int, γ::Float64)
    return tree.r[s_idx] .+ γ .* tree.v[s_idx]
end

node_return_sum(tree::SearchTree, s_idx::Int) = tree.return_sum[s_idx]

function add_return_sum!(tree::SearchTree, s_idx::Int, value::Float64)
    tree.return_sum[s_idx] += value
    return tree.return_sum[s_idx]
end

function selection_policy(::RegretMatchingSearch, tree::SearchTree, s_idx::Int; ϵ=0.30)
    x = regret_matching_policy(tree.regret[1][s_idx])
    y = regret_matching_policy(tree.regret[2][s_idx])
    return eps_exploration(x, ϵ), eps_exploration(y, ϵ)
end

function update_node!(::RegretMatchingSearch, tree::SearchTree, s_idx::Int, a::CartesianIndex{2}, total::Float64, π1, π2, γ::Float64)
    i, j = Tuple(a)
    q = node_matrix_game(tree, s_idx, γ)
    Δ1 = view(q, :, j) .- total
    Δ1[i] = 0.0
    tree.regret[1][s_idx] .+= Δ1
    Δ2 = total .- vec(view(q, i, :))
    Δ2[j] = 0.0
    tree.regret[2][s_idx] .+= Δ2
    tree.policy_sum[1][s_idx] .+= π1
    tree.policy_sum[2][s_idx] .+= π2
    return nothing
end

function zero_query_search(oracle, game::MG, s)
    x, y = state_policy(oracle, game, s)
    return Float64.(x), Float64.(y), oracle_state_value(oracle, game, s)
end

function search_info(params::MCTSSearch, game::MG, s; ϵ=0.30)
    tree = Tree(params, game, s)
    x, y, v = if isterminal(game, s)
        n1, n2 = length.(actions(game))
        uniform(n1), uniform(n2), 0.0
    elseif iszero(params.max_depth) || iszero(params.tree_queries) || iszero(params.max_time)
        zero_query_search(params.oracle, game, s)
    else
        start = time()
        for _ ∈ 1:params.tree_queries
            time() - start <= params.max_time || break
            simulate(params, tree, game, 1; ϵ)
        end
        search_result(params, tree, game, 1; ϵ)
    end
    return (x, y, v), (; tree)
end

search(params::MCTSSearch, game::MG, s; ϵ=0.30) =
    first(search_info(params, game, s; ϵ))

simulate(params::MCTSSearch, tree::SearchTree, game::MG, s_idx; ϵ=0.30) =
    simulate(params.search_style, params, tree, game, s_idx, 0; ϵ)

simulate(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30) =
    simulate_regret_matching(style, params, tree, game, s_idx, depth; ϵ)

function simulate_regret_matching(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30)
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

function search_result(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30)
    x, y = tree_policy(params, tree, game, s_idx; ϵ)
    v = node_value(params, tree, game, s_idx, x, y)
    return x, y, v
end

tree_policy(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30) =
    tree_policy(params.search_style, params, tree, game, s_idx; ϵ)

function tree_policy(::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30)
    if iszero(tree.n_s[s_idx]) || isempty(tree.r[s_idx])
        return oracle_policy(params, game, tree, s_idx)
    end
    x = normalize_or_uniform!(copy(tree.policy_sum[1][s_idx]))
    y = normalize_or_uniform!(copy(tree.policy_sum[2][s_idx]))
    return x, y
end

node_value(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, x, y) =
    node_value(params.search_style, params, tree, game, s_idx, x, y)

function node_value(::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, x, y)
    if iszero(tree.n_s[s_idx])
        return oracle_state_value(params.oracle, game, tree.s[s_idx])
    end
    return node_return_sum(tree, s_idx) / tree.n_s[s_idx]
end

function backup_value(style::RegretMatchingSearch, tree::SearchTree, s_idx::Int, sample_value::Float64)
    return style.backup == :mean ? node_return_sum(tree, s_idx) / tree.n_s[s_idx] : sample_value
end

function mcts_sim(params::MCTSSearch, game::MG, s; progress=false, ϵ=0.30, sim_depth::Int=params.max_depth)
    sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    rewards = Float64[]
    v_hist = Float64[]
    search_time_hist = Float64[]
    s_hist = Vector{Float32}[]
    policy_hist = (Vector{Float64}[], Vector{Float64}[])
    use_search_targets = params.value_target == :search
    if !use_search_targets && params.value_target != :rollout
        throw(ArgumentError("Unsupported value_target=$(params.value_target). Use :search or :rollout."))
    end
    p = Progress(sim_depth, enabled=progress)

    while (t <= sim_depth) && !isterminal(game, s)
        search_start = time()
        x, y, gv = search(params, game, s; ϵ)
        search_time = time() - search_start

        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(rewards, r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(v_hist, use_search_targets ? gv : 0.0)
        push!(policy_hist[1], x)
        push!(policy_hist[2], y)
        if !use_search_targets
            for _t ∈ eachindex(v_hist)
                v_hist[_t] += r * γ^(t - _t)
            end
        end
        t += 1
        s = sp
        next!(p)
    end
    if !use_search_targets && !isterminal(game, s)
        vp = oracle_state_value(params.oracle, game, s)
        for _t ∈ eachindex(v_hist)
            v_hist[_t] += vp * γ^(t - _t)
        end
    end
    finish!(p)
    return (;
        s = s_hist,
        r = rewards,
        v = v_hist,
        search_time = search_time_hist,
        policy = policy_hist,
    )
end
