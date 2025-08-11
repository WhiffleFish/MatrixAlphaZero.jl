function D3Trees.D3Tree(tree::Tree)
    ns = length(tree.s)
    γ = 0.95 # FIXME: Need this from the actual game
    children = vec.(tree.s_children)
    # children = [Int[] for _ in eachindex(tree.s)]
    # tooltip = Vector{String}(undef, ns)
    reach_probs = ones(length(tree.s))
    values = zeros(length(tree.s))
    for s_idx ∈ eachindex(tree.s)
        if is_leaf(tree, s_idx)
            values[s_idx] = 0.0
        else

        end
        x,y,t = solve(node_matrix_game(tree, 1.0, s_idx, γ))
        values[s_idx] = 
        for child ∈ tree.s_children[s_idx]
            reach_probs
        end
    end

    for s_idx ∈ eachindex(tree.s)
        r = round(tree.b_rewards[b_idx], sigdigits=3)
        tooltip[s_idx] = "s_idx = $s_idx\nr=$r"
    end

    return D3Trees.D3Tree(
        children;
        title = "MCTS Tree",
        # text,
        # style,
        # tooltip,
        # link_style,
    )
end
