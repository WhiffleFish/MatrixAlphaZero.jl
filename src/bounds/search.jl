function search_info(sol::BoundSolver, game::MG, s)
    tree = Tree(sol, game, s)
    (;max_iter, ϵ) = sol
    Q_lower_hist = Matrix{Float64}[]
    Q_upper_hist = Matrix{Float64}[]
    V̲ = Float64[]
    V̄ = Float64[]
    for i ∈ 1:max_iter
        simulate(sol, game, tree, 1, 0, value_gap(tree, 1)*ϵ)
        push!(Q_lower_hist, copy(tree.v_lower[1]))
        push!(Q_upper_hist, copy(tree.v_upper[1]))
        push!(V̲, tree.solved_vl[1])
        push!(V̄, tree.solved_vu[1])
    end
    return tree, (;
        Q̲ = Q_lower_hist,
        Q̄ = Q_upper_hist,
        V̲,
        V̄
    )
end

"""
Sample action according to upper bound policy
"""
function explore_action(tree::Tree, s_idx::Int)
    # CartesianIndex(
    #     rand(Categorical(tree.π̄[1][s_idx])), 
    #     rand(Categorical(tree.π̄[2][s_idx]))
    # )
    CartesianIndex(
        rand(Categorical((tree.π̲[1][s_idx] .+ tree.π̄[1][s_idx]) ./ 2)), 
        rand(Categorical((tree.π̲[2][s_idx] .+ tree.π̄[2][s_idx]) ./ 2))
    )
end


function simulate(sol::BoundSolver, game::MG, tree, s_idx, t, ϵ)
    (; max_depth) = sol
    # ϵ is trial improvement factor, not full algorithm termination condition
    γ = discount(game)
    s = tree.s[s_idx]

    if isterminal(game, s)
        return 0.0, 0.0
    elseif is_leaf(tree, s_idx) # TODO: keep exploring to decrease uncertainty
        expand_s!(tree, s_idx, game, sol)
    end
    if ϵ * γ^(-t) ≥ value_gap(tree, s_idx) # terminate
        return tree.solved_vl[s_idx], tree.solved_vu[s_idx]
    end

    a = explore_action(tree, s_idx)
    sp_idx = tree.s_children[s_idx][a]
    V̲sp , V̄sp = simulate(sol, game, tree, sp_idx, t+1, ϵ)

    tree.v_lower[s_idx][a] = V̲sp
    tree.v_upper[s_idx][a] = V̄sp
    
    x̲, y̲, V̲ = AZ.solve(lower_bound_matrix_game(tree, s_idx))
    x̄, ȳ, V̄ = AZ.solve(upper_bound_matrix_game(tree, s_idx))

    tree.π̲[1][s_idx] = x̲
    tree.π̲[2][s_idx] = y̲
    tree.π̄[1][s_idx] = x̄
    tree.π̄[2][s_idx] = ȳ

    tree.solved_vl[s_idx] = V̲
    tree.solved_vu[s_idx] = V̄

    return V̲, V̄
end

lower_bound_matrix_game(tree::Tree, s_idx::Int) = tree.r[s_idx] .+ discount(tree) .* tree.v_lower[s_idx]
upper_bound_matrix_game(tree::Tree, s_idx::Int) = tree.r[s_idx] .+ discount(tree) .* tree.v_upper[s_idx]

value_gap(tree::Tree, s_idx::Int) = tree.solved_vu[s_idx] - tree.solved_vl[s_idx]
