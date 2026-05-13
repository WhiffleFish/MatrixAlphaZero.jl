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

function regret_matching_policy(regret::AbstractVector)
    policy = zeros(Float64, length(regret))
    return match!(policy, regret)
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
    Δ1[i] = 0.0  # selected action contributes zero regret (paper: x(h,i,j)=u1 when (i,j) selected)
    tree.regret[1][s_idx] .+= Δ1
    Δ2 = total .- vec(view(q, i, :))
    Δ2[j] = 0.0
    tree.regret[2][s_idx] .+= Δ2
    tree.policy_sum[1][s_idx] .+= π1
    tree.policy_sum[2][s_idx] .+= π2
    return nothing
end

function tree_policy(style::RegretMatchingSearch, params::MCTSParams, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30)
    if iszero(tree.n_s[s_idx]) || isempty(tree.r[s_idx])
        return oracle_policy(params, game, tree, s_idx)
    end
    x = normalize_or_uniform!(copy(tree.policy_sum[1][s_idx]))
    y = normalize_or_uniform!(copy(tree.policy_sum[2][s_idx]))
    return x, y
end
