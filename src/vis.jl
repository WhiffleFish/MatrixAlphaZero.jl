function D3Trees.D3Tree(tree::SMOOSTree; title::String="SM-OOS Tree", kwargs...)
    ns = length(tree.s)
    children = [collect(values(tree.children[h])) for h in 1:ns]
    incoming = zeros(Int, ns)
    for h in 1:ns
        for child in children[h]
            incoming[child] = h
        end
    end

    text = Vector{String}(undef, ns)
    tooltip = Vector{String}(undef, ns)
    style = Vector{String}(undef, ns)
    link_style = fill("", ns)

    for h in 1:ns
        expanded = !isempty(tree.regret[1][h])
        child_count = length(children[h])
        r_norm = expanded ? sqrt(sum(abs2, tree.regret[1][h]) + sum(abs2, tree.regret[2][h])) : NaN
        s_mass = expanded ? sum(tree.strategy[1][h]) + sum(tree.strategy[2][h]) : NaN

        text[h] = join((
            "s$h",
            "R: $(isfinite(r_norm) ? _vis_fmt(r_norm) : "?")",
            "S: $(isfinite(s_mass) ? _vis_fmt(s_mass) : "?")",
        ), "\n")

        tooltip[h] = join((
            "state = s$h",
            "expanded = $expanded",
            "children = $child_count",
            "regret_norm = $(isfinite(r_norm) ? _vis_fmt(r_norm) : "?")",
            "strategy_mass = $(isfinite(s_mass) ? _vis_fmt(s_mass) : "?")",
            "parent = $(iszero(incoming[h]) ? "none" : "s$(incoming[h])")",
            "",
            repr(tree.s[h]),
        ), "\n")

        style[h] = expanded ? "fill:#e8f0fe;stroke:#2f5f9f" : "fill:#d9d9d9;stroke:#666"
        if h != 1
            link_style[h] = "stroke-width:2px;stroke-opacity:0.55"
        end
    end

    return D3Trees.D3Tree(
        children;
        text,
        tooltip,
        style,
        link_style,
        title,
        kwargs...,
    )
end

function _vis_fmt(x::Real)
    if abs(x) >= 100 || (0 < abs(x) < 1e-3)
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

    return if value >= 0
        "fill:rgb($(channel),245,$(channel));stroke:#2f6b2f"
    else
        "fill:rgb(245,$(channel),$(channel));stroke:#7a2f2f"
    end
end
