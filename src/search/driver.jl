function fitted_smoos_info(params::SMOOSParams, game::MG, s; ϵ=0.30)
    tree = Tree(params, game, s)
    if !isterminal(game, s)
        for _ ∈ 1:params.oos_iterations
            smoos_trajectory!(params, tree, game, 1, 0, 1.0, 1.0, 1.0, 1.0; ϵ)
        end
    end
    yr, ys = root_targets(params, tree, game, 1)
    return (yr, ys), (; tree)
end

fitted_smoos(params::SMOOSParams, game::MG, s; ϵ=0.30) =
    first(fitted_smoos_info(params, game, s; ϵ))

function smoos_trajectory!(
        params::SMOOSParams,
        tree::SMOOSTree,
        game::MG,
        h::Int,
        depth::Int,
        x1::Float64,
        x2::Float64,
        q1::Float64,
        q2::Float64;
        ϵ=0.30
    )
    s = tree.s[h]
    if isterminal(game, s)
        return 0.0
    elseif depth >= params.max_depth
        return oracle_state_value(params.oracle, game, s)
    end

    expand_node!(tree, h, game, params)
    σ1 = regret_matching_policy(tree.regret[1][h])
    σ2 = regret_matching_policy(tree.regret[2][h])
    σp1 = eps_exploration(σ1, ϵ)
    σp2 = eps_exploration(σ2, ϵ)

    a = action_idx_from_probs(σp1, σp2)
    i, j = Tuple(a)
    A1, A2 = actions(game)
    sp, r = @gen(:sp, :r)(game, s, (A1[i], A2[j]))
    r = zs_reward_scalar(r)
    hp = child_index!(tree, h, a, sp)

    g′ = smoos_trajectory!(
        params, tree, game, hp, depth + 1,
        x1 * σ1[i], x2 * σ2[j],
        q1 * σp1[i], q2 * σp2[j];
        ϵ
    )
    g = Float64(r) + discount(game) * g′

    qreach = max(q1 * q2, eps(Float64))
    f1 = x2 / qreach
    f2 = x1 / qreach
    for b ∈ eachindex(σ1)
        tree.regret[1][h][b] += f1 * ((b == i ? 1.0 : 0.0) - σ1[b]) * g
    end
    for b ∈ eachindex(σ2)
        tree.regret[2][h][b] += f2 * (σ2[b] - (b == j ? 1.0 : 0.0)) * g
    end

    ω1 = x1 / max(qreach, eps(Float64))
    ω2 = x2 / max(qreach, eps(Float64))
    tree.strategy[1][h] .+= ω1 .* σ1
    tree.strategy[2][h] .+= ω2 .* σ2
    return g
end
