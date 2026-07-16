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
    fresh_regret:: NTuple{2, Vector{Vector{Float64}}}
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
        ([NO_FLOAT], [NO_FLOAT]),
    )
end

Tree(::MCTSSearch, game::MG, s=rand(initialstate(game))) = SearchTree(game, s)

is_leaf(tree::SearchTree, s_idx::Int) = isempty(tree.s_children[s_idx])

function reset_search_node!(tree::SearchTree, s_idx::Int, na1::Int, na2::Int)
    tree.return_sum[s_idx] = 0.0
    tree.regret[1][s_idx] = zeros(Float64, na1)
    tree.regret[2][s_idx] = zeros(Float64, na2)
    tree.fresh_regret[1][s_idx] = zeros(Float64, na1)
    tree.fresh_regret[2][s_idx] = zeros(Float64, na2)
    tree.policy_sum[1][s_idx] = zeros(Float64, na1)
    tree.policy_sum[2][s_idx] = zeros(Float64, na2)
    return nothing
end

function append_search_frontier!(tree::SearchTree, n_frontier::Int)
    append!(tree.return_sum, fill(0.0, n_frontier))
    foreach(tree.regret) do regret_i
        append!(regret_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.fresh_regret) do regret_i
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

function has_regret_transfer(params::MCTSSearch)
    if uses_loss_scaled_transfer(params)
        regret_mass, strategy_mass = transfer_pseudo_masses(params)
        return regret_mass > 0 || strategy_mass > 0
    end
    return params.transfer_weight > 0 && params.τ > 0
end

# Warm-start init injected into a node's regret/policy_sum accumulators, mirroring
# SM-OOS `transfer_prior`. RM+ clips the transferred accumulator to its feasible
# nonnegative state before search begins.
function mcts_transfer_prior(params::MCTSSearch, game::MG, s, na1::Int, na2::Int, learned_reach::Real=1.0)
    r̂ = state_regret(params.oracle, game, s)
    ŝ = state_strategy(params.oracle, game, s)
    if uses_loss_scaled_transfer(params)
        regret_pseudo_mass, strategy_pseudo_mass = transfer_pseudo_masses(params, learned_reach)
        regret_mass = sqrt(regret_pseudo_mass)
        projection_mass = regret_pseudo_mass
    else
        regret_mass = sqrt(params.transfer_weight * params.τ)
        projection_mass = params.τ
        strategy_pseudo_mass = params.τ
    end
    Δ = params.transfer_payoff_bound
    r1 = project_transfer_regret!(regret_mass .* Float64.(r̂[1]), projection_mass, na1, Δ)
    r2 = project_transfer_regret!(regret_mass .* Float64.(r̂[2]), projection_mass, na2, Δ)
    prepare_transfer_regret!(params.search_style.method, r1)
    prepare_transfer_regret!(params.search_style.method, r2)
    s1 = strategy_pseudo_mass .* normalized_or_uniform(Float64.(ŝ[1]))
    s2 = strategy_pseudo_mass .* normalized_or_uniform(Float64.(ŝ[2]))
    return (r1, r2), (s1, s2)
end

prepare_transfer_regret!(::Vanilla, regret) = regret

function prepare_transfer_regret!(::Plus, regret)
    regret .= max.(regret, 0.0)
    return regret
end

function warmstart_node!(params::MCTSSearch, tree::SearchTree, s_idx::Int, game::MG; learned_reach::Real=1.0)
    has_regret_transfer(params) || return nothing
    na1, na2 = size(tree.n_sa[s_idx])
    (r1, r2), (s1, s2) = mcts_transfer_prior(params, game, tree.s[s_idx], na1, na2, learned_reach)
    tree.regret[1][s_idx] .= r1
    tree.regret[2][s_idx] .= r2
    tree.policy_sum[1][s_idx] .= s1
    tree.policy_sum[2][s_idx] .= s2
    return nothing
end

# Vanilla decontaminates regret targets by subtracting its additive prior. RM+
# uses a shadow accumulator that applies the same fresh updates from zero because
# its per-update clamp makes subtraction invalid.
decontaminated_regret(::Vanilla, regret, fresh_regret, prior) = regret .- prior
decontaminated_regret(::Plus, regret, fresh_regret, prior) = fresh_regret

function mcts_root_targets(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int)
    A1, A2 = actions(game)
    na1, na2 = length(A1), length(A2)
    if has_regret_transfer(params)
        (r1i, r2i), (s1i, s2i) = mcts_transfer_prior(params, game, tree.s[s_idx], na1, na2)
    else
        r1i, r2i = zeros(na1), zeros(na2)
        s1i, s2i = zeros(na1), zeros(na2)
    end
    T = tree.n_s[s_idx]
    regret_scale_iterations = uses_loss_scaled_transfer(params) ? T : params.τ + T
    regret_denom = sqrt(max(Float64(regret_scale_iterations), 1.0))
    yr = (
        Float64.(decontaminated_regret(
            params.search_style.method,
            tree.regret[1][s_idx],
            tree.fresh_regret[1][s_idx],
            r1i,
        )) ./ regret_denom,
        Float64.(decontaminated_regret(
            params.search_style.method,
            tree.regret[2][s_idx],
            tree.fresh_regret[2][s_idx],
            r2i,
        )) ./ regret_denom,
    )
    ys = (
        normalized_or_uniform(max.(Float64.(tree.policy_sum[1][s_idx]) .- s1i, 0.0)),
        normalized_or_uniform(max.(Float64.(tree.policy_sum[2][s_idx]) .- s2i, 0.0)),
    )
    return yr, ys
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

function accumulate_regret!(::Vanilla, regret, delta)
    regret .+= delta
    return regret
end

function accumulate_regret!(::Plus, regret, delta)
    regret .= max.(regret .+ delta, 0.0)
    return regret
end

function update_node!(style::RegretMatchingSearch, tree::SearchTree, s_idx::Int, a::CartesianIndex{2}, total::Float64, π1, π2, γ::Float64)
    i, j = Tuple(a)
    q = node_matrix_game(tree, s_idx, γ)
    Δ1 = view(q, :, j) .- total
    Δ1[i] = 0.0
    accumulate_regret!(style.method, tree.regret[1][s_idx], Δ1)
    accumulate_regret!(style.method, tree.fresh_regret[1][s_idx], Δ1)
    Δ2 = total .- vec(view(q, i, :))
    Δ2[j] = 0.0
    accumulate_regret!(style.method, tree.regret[2][s_idx], Δ2)
    accumulate_regret!(style.method, tree.fresh_regret[2][s_idx], Δ2)
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

simulate(params::MCTSSearch, tree::SearchTree, game::MG, s_idx; ϵ=0.30, learned_reach::Float64=1.0) =
    simulate(params.search_style, params, tree, game, s_idx, 0; ϵ, learned_reach)

simulate(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30, learned_reach::Float64=1.0) =
    simulate_regret_matching(style, params, tree, game, s_idx, depth; ϵ, learned_reach)

function simulate_regret_matching(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30, learned_reach::Float64=1.0)
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
        warmstart_node!(params, tree, s_idx, game; learned_reach)
        leaf_value = oracle_state_value(params.oracle, game, s)
        add_return_sum!(tree, s_idx, leaf_value)
        tree.n_s[s_idx] += 1
        return leaf_value
    else
        γ = discount(game)
        π1, π2 = selection_policy(style, tree, s_idx; ϵ)
        a = action_idx_from_probs(π1, π2)
        sp_idx = tree.s_children[s_idx][a]
        i, j = Tuple(a)
        child_learned_reach = learned_reach * tree.prior[1][s_idx][i] * tree.prior[2][s_idx][j]
        vp = simulate(style, params, tree, game, sp_idx, depth + 1; ϵ, learned_reach=child_learned_reach)
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

# Self-play for RegretMatchingSearch with a FittedRegretModel oracle: emits
# decontaminated regret/strategy targets (like smoos_sim) but with exact,
# importance-sampling-free regret estimates from the MCTS backup.
function mcts_regret_sim(params::MCTSSearch, game::MG, s; progress=false, ϵ=0.30, sim_depth::Int=params.max_depth, gae_lambda=0.95)
    sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    use_search_targets = params.value_target == :search
    if !use_search_targets && params.value_target != :gae
        throw(ArgumentError("Unsupported value_target=$(params.value_target) for fitted-regret MCTS self-play. Use :search or :gae."))
    end
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
        (_x, _y, gv), info = search_info(params, game, s; ϵ)
        yr, ys = mcts_root_targets(params, info.tree, game, 1)
        search_time = time() - search_start

        x = eps_exploration(normalized_or_uniform(ys[1]), ϵ)
        y = eps_exploration(normalized_or_uniform(ys[2]), ϵ)
        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(values, use_search_targets ? gv : oracle_state_value(params.oracle, game, s))
        push!(rewards, Float64(r))
        push!(regret_hist[1], Float64.(yr[1]))
        push!(regret_hist[2], Float64.(yr[2]))
        push!(strategy_hist[1], Float64.(ys[1]))
        push!(strategy_hist[2], Float64.(ys[2]))
        t += 1
        s = sp
        next!(p)
    end
    v_hist = if use_search_targets
        values
    else
        bootstrap = isterminal(game, s) ? 0.0 : oracle_state_value(params.oracle, game, s)
        lambda_gae_targets(rewards, values, bootstrap, γ, gae_lambda)
    end
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
