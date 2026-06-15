using Base.Threads: @threads
using Random

function policy1_from_tree(game, planner, tree)
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.strategy[1][idx])
            x, _ = AZ.state_strategy(AZ.oracle(planner), game, s)
            return x
        else
            return AZ.normalized_or_uniform(tree.strategy[1][idx])
        end
    end
end

function policy2_from_tree(game, planner, tree)
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.strategy[2][idx])
            _, y = AZ.state_strategy(AZ.oracle(planner), game, s)
            return y
        else
            return AZ.normalized_or_uniform(tree.strategy[2][idx])
        end
    end
end

function joint_policy_from_tree(game, planner, tree)
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.strategy[1][idx])
            return AZ.state_strategy(AZ.oracle(planner), game, s)
        else
            return AZ.normalized_or_uniform(tree.strategy[1][idx]), AZ.normalized_or_uniform(tree.strategy[2][idx])
        end
    end
end

function search_eval(planner::AlphaZeroPlanner, params::SMOOSSearch, game::MG, s; ϵ=0.30, every=10, progress=true)
    tree = AZ.Tree(params, game, s)
    (;oos_iterations) = params
    iter    = Int[]
    brvs1   = Float64[]
    brvs2   = Float64[]
    v       = Float64[]
    p = Progress(oos_iterations; enabled=progress)
    for i ∈ 1:oos_iterations
        AZ.smoos_trajectory!(params, tree, game, 1, 0, 1.0, 1.0, 1.0, 1.0; ϵ)
        if iszero(mod(i, every))
            π1 = policy1_from_tree(game, planner, tree)
            π2 = policy2_from_tree(game, planner, tree)
            planner_oracle = AZ.oracle(planner)
            brv1, brv2 = approx_br_values_both_st(game, planner_oracle, π1, π2, s; max_depth=5)
            t = AZ.oracle_state_value(planner_oracle, game, s)
            push!(iter, i)
            push!(brvs1, brv1)
            push!(brvs2, brv2)
            push!(v, t)
        end
        next!(p)
    end
    finish!(p)
    return (;
        iter, 
        brv1 = brvs1, 
        brv2 = brvs2, 
        v
    )
end

struct FunctionBehavior{F} <: Policy
    f::F
end

MarkovGames.behavior(pol::FunctionBehavior, s) = pol.f(s)

function exploit_p1_joint_policy(planner::AlphaZeroPlanner, game::MG)
    return function (s)
        σ, info = behavior_info(planner, s)
        π1 = policy1_from_tree(game, planner, info.tree)
        π2 = policy2_from_tree(game, planner, info.tree)
        brv1, brv2, br2_map, br1_map = approx_br_values_both_st(game, AZ.oracle(planner), π1, π2, s; max_depth=5, return_policies=true)
        a2 = br2_map[(s,0)]
        return ProductDistribution(σ[1], Deterministic(actions(game)[2][a2]))
    end |> FunctionBehavior
end

function exploit_p2_joint_policy(planner::AlphaZeroPlanner, game::MG)
    return function (s)
        σ, info = behavior_info(planner, s)
        π1 = policy1_from_tree(game, planner, info.tree)
        π2 = policy2_from_tree(game, planner, info.tree)
        brv1, brv2, br2_map, br1_map = approx_br_values_both_st(game, AZ.oracle(planner), π1, π2, s; max_depth=5, return_policies=true)
        a1 = br1_map[(s,0)]
        return ProductDistribution(Deterministic(actions(game)[2][a1]), σ[2])
    end |> FunctionBehavior
end

sim_eval(planner::AlphaZeroPlanner, params::SMOOSSearch, game::MG, s; kwargs...) =
    search_eval(planner, params, game, s; kwargs...)


# Convenience: policies from your oracle (policy head)
policy1_from_oracle(oracle) = (game, s) -> first(AZ.state_strategy(oracle, game, s))
policy2_from_oracle(oracle) = (game, s) -> last(AZ.state_strategy(oracle, game, s))


##
############################ Single-traversal LLBR (both directions) ############################

struct LLBRBothCtx{G,O,VO,P1,P2,SF,S}
    game::G
    oracle::O
    value_oracle::VO
    π1::P1                       # (game, s) -> probs over A1
    π2::P2                       # (game, s) -> probs over A2
    γ::Float64
    max_depth::Int
    cache::Dict{Tuple{UInt64,Int},NTuple{2,Float64}}  # (state_key, depth) -> (v1_br2, v2_br1)
    br2_map::Dict{Tuple{S,Int},Int}              # argmin a2 for v1_br2 at (s,d)
    br1_map::Dict{Tuple{S,Int},Int}              # argmax a1 for v2_br1 at (s,d)
    record_policies::Bool
    statefeat::SF                                     # (s, game) -> Vector{Float32}
end

@inline state_key(ctx::LLBRBothCtx, s) = hash(ctx.statefeat(s, ctx.game))

@inline function oracle_val_p1(ctx::LLBRBothCtx, s)
    sv = ctx.statefeat(s, ctx.game)
    return Float64(only(AZ.value(ctx.value_oracle, sv)))
end

# Single recursion that returns both values: (v1_br2, v2_br1) from P1's perspective and then negated.
function V(ctx::LLBRBothCtx, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return (0.0, 0.0)
    elseif d ≥ ctx.max_depth
        v = oracle_val_p1(ctx, s)               # bootstrap is P1's value
        return (v, v)                          # both recursions share same bootstrap
    end
    if cache
        key = (state_key(ctx, s), d)
        if (v = get(ctx.cache, key, (NaN, NaN)); !isnan(v[1]))
            return v
        end
    end

    (; γ, game) = ctx
    A1, A2 = actions(ctx.game)
    π1 = ctx.π1(ctx.game, s)
    π2 = ctx.π2(ctx.game, s)
    n1, n2 = length(A1), length(A2)

    q12 = zeros(n2)  # for v1_br2: one value per a2 (we'll take min)
    q21 = zeros(n1)  # for v2_br1: one value per a1 (we'll take max)

    # Visit each joint action once; reuse both children values.
    @inbounds for j in 1:n2
        a2 = A2[j]
        for i in 1:n1
            a1 = A1[i]
            sp, r = @gen(:sp, :r)(game, s, (a1, a2))
            r1 = AZ.zs_reward_scalar(r)                  # P1 reward (zero-sum)
            v12p, v21p = V(ctx, sp, d + 1)               # single recursive call returns both
            q12[j] += π1[i] * (r1 + γ * v12p)        # expectation over π1 for fixed a2
            q21[i] += π2[j] * (r1 + γ * v21p)        # expectation over π2 for fixed a1
        end
    end

    # Best responses
    jmin = argmin(q12); v1_br2 = q12[jmin]               # P2 BR to π1 ⇒ minimize P1 value
    imax = argmax(q21); v2_br1 = q21[imax]               # P1 BR to π2 ⇒ maximize P1 value

    if ctx.record_policies
        key = (s, d)
        ctx.br2_map[key] = jmin
        ctx.br1_map[key] = imax
    end
    if cache
        ctx.cache[key] = (v1_br2, v2_br1)
    end
    return (v1_br2, v2_br1)
end

"""
    approx_br_values_both_st(game::MG, oracle, π1, π2, s;
                             max_depth::Int=5, return_policies::Bool=false)

Compute BOTH limited-lookahead best-response utilities in a SINGLE traversal.

Returns `(v1_br2, v2_br1)` where:
- `v1_br2`: P1 value when P2 plays BR to π₁ (min over a₂, expect over π₁)
- `v2_br1`: P2 value when P1 plays BR to π₂ (max over a₁, expect over π₂)

If `return_policies=true`, returns `(v1_br2, v2_br1, br2_map, br1_map)`, where the maps
store the chosen BR action index at each visited `(state_hash, depth)`.
"""
function approx_br_values_both_st(game::MG{S}, oracle, π1, π2, s;
                                  max_depth::Int=5, return_policies::Bool=false, cache=false, value_oracle=oracle) where S
    ctx = LLBRBothCtx(
        game, oracle, value_oracle, π1, π2,
        discount(game), max_depth,
        Dict{Tuple{UInt64,Int},NTuple{2,Float64}}(),
        Dict{Tuple{S,Int},Int}(),
        Dict{Tuple{S,Int},Int}(),
        return_policies,
        (s_, g_) -> MarkovGames.convert_s(Vector{Float32}, s_, g_),
    )
    v1_br2, v2_br1 = V(ctx, s, 0; cache)
    return return_policies ? (v1_br2, -v2_br1, ctx.br2_map, ctx.br1_map) : (v1_br2, -v2_br1)
end
