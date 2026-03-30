struct Exp3Tree{S} <: AbstractBanditTree
    core        :: SearchTreeCore{S}
    return_sum  :: Vector{Float64}
    reward_sum  :: NTuple{2, Vector{Vector{Float64}}}
end

function Exp3Tree(game::MG, s=rand(initialstate(game)))
    return Exp3Tree(
        SearchTreeCore(game, s),
        [0.0],
        ([NO_FLOAT], [NO_FLOAT]),
    )
end

Tree(::Exp3Search, game::MG, s=rand(initialstate(game))) = Exp3Tree(game, s)

function reset_search_node!(tree::Exp3Tree, s_idx::Int, na1::Int, na2::Int)
    tree.return_sum[s_idx] = 0.0
    tree.reward_sum[1][s_idx] = zeros(Float64, na1)
    tree.reward_sum[2][s_idx] = zeros(Float64, na2)
    return nothing
end

function append_search_frontier!(tree::Exp3Tree, n_frontier::Int)
    append!(tree.return_sum, fill(0.0, n_frontier))
    foreach(tree.reward_sum) do reward_i
        append!(reward_i, fill(NO_FLOAT, n_frontier))
    end
    return nothing
end

function exp3_policy(reward::AbstractVector, style::Exp3Search; ϵ=0.30)
    n = length(reward)
    η = isnan(style.η) ? max(ϵ, inv(n)) / n : style.η
    shifted = reward .- maximum(reward)
    logits = exp.(η .* shifted)
    normalize_or_uniform!(logits)
    return eps_exploration(logits, ϵ)
end

function selection_policy(style::Exp3Search, tree::Exp3Tree, s_idx::Int; ϵ=0.30)
    x = exp3_policy(tree.reward_sum[1][s_idx], style; ϵ)
    y = exp3_policy(tree.reward_sum[2][s_idx], style; ϵ)
    return x, y
end

function update_node!(::Exp3Search, tree::Exp3Tree, s_idx::Int, a::CartesianIndex{2}, total::Float64, π1, π2, γ::Float64)
    i, j = Tuple(a)
    tree.reward_sum[1][s_idx][i] += total / max(π1[i], eps(Float64))
    tree.reward_sum[2][s_idx][j] += (-total) / max(π2[j], eps(Float64))
    return nothing
end

function tree_policy(style::Exp3Search, params::MCTSParams, tree::Exp3Tree, game::MG, s_idx::Int; ϵ=0.30)
    if iszero(tree.n_s[s_idx]) || isempty(tree.r[s_idx])
        return oracle_policy(params, game, tree, s_idx)
    elseif style.target_policy == :empirical
        return empirical_policy(tree, s_idx)
    else
        return selection_policy(style, tree, s_idx; ϵ)
    end
end
