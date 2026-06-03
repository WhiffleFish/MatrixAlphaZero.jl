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
        return 1.0, 1.0, 1.0, 1.0, 0.0
    elseif depth ≥ params.max_depth
        return 1.0, 1.0, 1.0, 1.0, oracle_state_value(params.oracle, game, s)
    end

    expand_node!(tree, h, game, params)
    σ1 = regret_matching_policy(tree.regret[1][h])
    σ2 = regret_matching_policy(tree.regret[2][h])
    σ1_sample = eps_exploration(σ1, ϵ)
    σ2_sample = eps_exploration(σ2, ϵ)

    a = action_idx_from_probs(σ1_sample, σ2_sample)
    i, j = Tuple(a)
    A1, A2 = actions(game)
    sp, r = @gen(:sp, :r)(game, s, (A1[i], A2[j]))
    r = zs_reward_scalar(r)

    hp = child_index!(tree, h, a, sp)
    tail_x1, tail_x2, tail_q1, tail_q2, tail_value = smoos_trajectory!(
        params, tree, game, hp, depth + 1,
        x1 * σ1[i], x2 * σ2[j],
        q1 * σ1_sample[i], q2 * σ2_sample[j];
        ϵ
    )

    value = Float64(r) + discount(game) * tail_value
    sample_reach = max(
        q1 * q2 * σ1_sample[i] * σ2_sample[j] * tail_q1 * tail_q2,
        eps(Float64),
    )
    w1 = value * (x2 * σ2[j] * tail_x2) * tail_x1 / sample_reach
    w2 = -value * (x1 * σ1[i] * tail_x1) * tail_x2 / sample_reach

    for b ∈ eachindex(σ1)
        tree.regret[1][h][b] += ((b == i) - σ1[b]) * w1
    end
    for b ∈ eachindex(σ2)
        tree.regret[2][h][b] += ((b == j) - σ2[b]) * w2
    end
    tree.strategy[1][h] .+= σ1
    tree.strategy[2][h] .+= σ2
    return tail_x1 * σ1[i], tail_x2 * σ2[j], tail_q1 * σ1_sample[i], tail_q2 * σ2_sample[j], value
end
