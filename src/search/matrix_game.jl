struct MatrixGameTree{S} <: AbstractSearchTree
    core::SearchTreeCore{S}
end

MatrixGameTree(game::MG, s=rand(initialstate(game))) = MatrixGameTree(SearchTreeCore(game, s))

Tree(::MatrixGameSearch, game::MG, s=rand(initialstate(game))) = MatrixGameTree(game, s)

reset_search_node!(tree::MatrixGameTree, s_idx::Int, na1::Int, na2::Int) = nothing
append_search_frontier!(tree::MatrixGameTree, n_frontier::Int) = nothing

function simulate(::MatrixGameSearch, params::MCTSParams, tree::MatrixGameTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30)
    γ = discount(game)
    s = tree.s[s_idx]

    if isterminal(game, s)
        return 0.0
    elseif is_leaf(tree, s_idx)
        expand_s!(tree, s_idx, game, params.oracle)
        x, y, t = solve(params.matrix_solver, node_matrix_game(tree, params.c, s_idx, γ))
        return t
    else
        a = explore_action(params.matrix_solver, tree, params.c, s_idx, γ; ϵ)
        sp_idx = tree.s_children[s_idx][a]
        vp = simulate(MatrixGameSearch(), params, tree, game, sp_idx, depth + 1; ϵ)

        v̂ = tree.v[s_idx][a]
        nsa = tree.n_sa[s_idx][a]
        tree.v[s_idx][a] = v̂ + (vp - v̂) / (nsa + 1)
        tree.n_s[s_idx] += 1
        tree.n_sa[s_idx][a] += 1

        x, y, t = solve(params.matrix_solver, node_matrix_game(tree, params.c, s_idx, γ))
        return t
    end
end

function tree_policy(::MatrixGameSearch, params::MCTSParams, tree::MatrixGameTree, game::MG, s_idx::Int; ϵ=0.30)
    γ = discount(game)
    x, y, _ = solve(params.matrix_solver, node_matrix_game(tree, params.c, s_idx, γ))
    return x, y
end

function node_value(::MatrixGameSearch, params::MCTSParams, tree::MatrixGameTree, game::MG, s_idx::Int, x, y)
    γ = discount(game)
    return dot(x, node_matrix_game(tree, s_idx, γ), y)
end
