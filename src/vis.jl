function D3Trees.D3Tree(tree::AbstractSearchTree; γ::Float64=0.95, title::String="MCTS Tree", kwargs...)
    ns = length(tree.s)
    children = [collect(vec(s_children)) for s_children ∈ tree.s_children]

    values = fill(NaN, ns)
    reach_probs = zeros(Float64, ns)
    reach_probs[1] = 1.0

    incoming_visits = zeros(Int, ns)
    incoming_policy = zeros(Float64, ns)
    incoming_reward = fill(NaN, ns)

    for s_idx ∈ eachindex(tree.s)
        if isempty(tree.r[s_idx])
            continue
        end

        x, y, values[s_idx] = solve(node_matrix_game(tree, s_idx, γ))
        row_policy = repeat(x, outer=length(y))
        col_policy = repeat(y, inner=length(x))

        child_ids = children[s_idx]
        counts = vec(tree.n_sa[s_idx])
        rewards = vec(tree.r[s_idx])

        for (edge_idx, child_idx) ∈ enumerate(child_ids)
            joint_prob = row_policy[edge_idx] * col_policy[edge_idx]
            reach_probs[child_idx] = reach_probs[s_idx] * joint_prob
            incoming_visits[child_idx] = counts[edge_idx]
            incoming_policy[child_idx] = joint_prob
            incoming_reward[child_idx] = rewards[edge_idx]
        end
    end

    node_visits = [
        max(
            tree.n_s[s_idx],
            incoming_visits[s_idx],
            Int(!isempty(tree.r[s_idx]))
        ) for s_idx ∈ 1:ns
    ]

    text = Vector{String}(undef, ns)
    tooltip = Vector{String}(undef, ns)
    style = Vector{String}(undef, ns)
    link_style = fill("", ns)

    finite_values = filter(isfinite, values)
    max_abs_value = isempty(finite_values) ? 1.0 : max(maximum(abs, finite_values), eps())
    max_edge_visits = max(maximum(incoming_visits), 1)

    for s_idx ∈ 1:ns
        expanded = !isempty(tree.r[s_idx])
        leaf = is_leaf(tree, s_idx)
        child_count = length(children[s_idx])
        value_txt = isfinite(values[s_idx]) ? _vis_fmt(values[s_idx]) : "?"
        reach_txt = _vis_fmt(reach_probs[s_idx])

        text[s_idx] = join((
            "s$s_idx",
            "V: $value_txt",
            "N: $(node_visits[s_idx])",
        ), "\n")

        tooltip[s_idx] = join((
            "state = s$s_idx",
            "expanded = $expanded",
            "leaf = $leaf",
            "children = $child_count",
            "value = $value_txt",
            "reach = $reach_txt",
            "edge_policy = $(_vis_fmt(incoming_policy[s_idx]))",
            "edge_visits = $(incoming_visits[s_idx])",
            "edge_reward = $(isfinite(incoming_reward[s_idx]) ? _vis_fmt(incoming_reward[s_idx]) : "?")",
            "",
            repr(tree.s[s_idx]),
        ), "\n")

        style[s_idx] = _value_style(values[s_idx], max_abs_value)

        if s_idx != 1
            width = 1 + 11 * sqrt(incoming_visits[s_idx] / max_edge_visits)
            opacity = 0.25 + 0.75 * max(incoming_policy[s_idx], reach_probs[s_idx])
            link_style[s_idx] = "stroke-width:$(_vis_fmt(width))px;stroke-opacity:$(_vis_fmt(min(opacity, 1.0)))"
        end
    end

    return D3Trees.D3Tree(
        children;
        text=text,
        tooltip=tooltip,
        style=style,
        link_style=link_style,
        title=title,
        kwargs...,
    )
end

function _vis_fmt(x::Real)
    if abs(x) ≥ 100 || (0 < abs(x) < 1e-3)
        return string(round(x; sigdigits=3))
    end
    return string(round(x; digits=3))
end

function _value_style(value::Real, max_abs_value::Float64)
    if !isfinite(value)
        return "fill:#d9d9d9;stroke:#666"
    end

    scale = clamp(abs(value) / max_abs_value, 0.0, 1.0)
    lo = 245
    hi = 70
    channel = round(Int, lo - (lo - hi) * scale)

    return if value ≥ 0
        "fill:rgb($(channel),245,$(channel));stroke:#2f6b2f"
    else
        "fill:rgb(245,$(channel),$(channel));stroke:#7a2f2f"
    end
end
