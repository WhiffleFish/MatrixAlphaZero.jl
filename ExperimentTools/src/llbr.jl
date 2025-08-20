############ Multithreaded Limited-Lookahead Best Response to P1 ############

using Base.Threads: @threads
using Random

# Concrete context to avoid captured vars
struct BRContext{G,O,P,SF}
    game::G
    oracle::O
    policy1::P
    γ::Float64
    max_depth::Int
    parallel_depth::Int
    oracle_threadsafe::Bool
    cache::Dict{Tuple{UInt64,Int},Float64}
    br_policy::Dict{Tuple{UInt64,Int},Int}  # (state_key, depth) -> best_j
    cache_lock::ReentrantLock
    br_lock::ReentrantLock
    oracle_lock::ReentrantLock
    statefeat::SF  # (s, game) -> Vector{Float32}
end

@inline state_key(ctx::BRContext, s) = hash(ctx.statefeat(s, ctx.game))

function oracle_val(ctx::BRContext, s)
    z = ctx.statefeat(s, ctx.game)
    if ctx.oracle_threadsafe
        return Float64(only(AZ.value(ctx.oracle, z)))
    else
        lock(ctx.oracle_lock)
        v = only(AZ.value(ctx.oracle, z))
        unlock(ctx.oracle_lock)
        return Float64(v)
    end
end

# NO inner function; top-level, so nothing is captured
function V(ctx::BRContext, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return 0.0
    elseif d ≥ ctx.max_depth
        return oracle_val(ctx, s)
    end

    key = (state_key(ctx, s), d)

    # cache read
    if cache
        lock(ctx.cache_lock)
        cached = get(ctx.cache, key, NaN)
        unlock(ctx.cache_lock)
        if !isnan(cached)
            return cached
        end
    end

    A1, A2 = actions(ctx.game)
    π1 = ctx.policy1(ctx.game, s)
    m = length(A2)
    vals = Vector{Float64}(undef, m)

    if d < ctx.parallel_depth && m > 1
        @threads for j in 1:m
            a2 = A2[j]
            q = 0.0
            @inbounds for (i, a1) in enumerate(A1)
                sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
                r1 = AZ.zs_reward_scalar(r)
                q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
            end
            vals[j] = q
        end
    else
        @inbounds for j in 1:m
            a2 = A2[j]
            q = 0.0
            for (i, a1) in enumerate(A1)
                sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
                r1 = AZ.zs_reward_scalar(r)
                q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
            end
            vals[j] = q
        end
    end

    best_j = argmin(vals)
    best_val = vals[best_j]

    # record BR action
    lock(ctx.br_lock); ctx.br_policy[key] = best_j; unlock(ctx.br_lock)

    # cache write (double-check)
    if cache
        lock(ctx.cache_lock)
        if !haskey(ctx.cache, key)
            ctx.cache[key] = best_val
        else
            best_val = ctx.cache[key]
        end
        unlock(ctx.cache_lock)
    end

    return best_val
end

function Vs(ctx::BRContext, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return 0.0, 0.0
    elseif d ≥ ctx.max_depth
        v̂ = oracle_val(ctx, s)
        return v̂, -v̂
    end

    key = (state_key(ctx, s), d)

    # cache read
    if cache
        lock(ctx.cache_lock)
        cached = get(ctx.cache, key, NaN)
        unlock(ctx.cache_lock)
        if !isnan(cached)
            return cached
        end
    end

    A1, A2 = actions(ctx.game)
    π1, π2 = ctx.policy1(ctx.game, s)
    m = length(A2)
    vals = Vector{Float64}(undef, m)

    if d < ctx.parallel_depth && m > 1
        @threads for j in 1:m
            a2 = A2[j]
            q = 0.0
            @inbounds for (i, a1) in enumerate(A1)
                sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
                r1 = AZ.zs_reward_scalar(r)
                q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
            end
            vals[j] = q
        end
    else
        @inbounds for j in 1:m
            a2 = A2[j]
            q = 0.0
            for (i, a1) in enumerate(A1)
                sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
                r1 = AZ.zs_reward_scalar(r)
                q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
            end
            vals[j] = q
        end
    end

    best_j = argmin(vals)
    best_val = vals[best_j]

    # record BR action
    lock(ctx.br_lock); ctx.br_policy[key] = best_j; unlock(ctx.br_lock)

    # cache write (double-check)
    if cache
        lock(ctx.cache_lock)
        if !haskey(ctx.cache, key)
            ctx.cache[key] = best_val
        else
            best_val = ctx.cache[key]
        end
        unlock(ctx.cache_lock)
    end

    return best_val
end

"""
    approx_br_value_mt(game::MG, oracle, policy1, s;
                       max_depth::Int=5,
                       return_policy::Bool=false,
                       parallel_depth::Int=2,
                       oracle_threadsafe::Bool=true)

Multithreaded limited-lookahead BR to P1 without captured variables.
Returns P1's value at the root (and optionally the BR choices map).
"""
function approx_br_value_mt(game::MG, oracle, policy1, s;
                            max_depth::Int=5,
                            return_policy::Bool=false,
                            parallel_depth::Int=2,
                            oracle_threadsafe::Bool=true)

    ctx = BRContext(
        game,
        oracle,
        policy1,
        discount(game),
        max_depth,
        parallel_depth,
        oracle_threadsafe,
        Dict{Tuple{UInt64,Int},Float64}(),
        Dict{Tuple{UInt64,Int},Int}(),
        ReentrantLock(),
        ReentrantLock(),
        ReentrantLock(),
        (s_, g_) -> MarkovGames.convert_s(Vector{Float32}, s_, g_),
    )

    root_val = V(ctx, s, 0)
    return return_policy ? (root_val, ctx.br_policy) : root_val
end


############ Limited-Lookahead Best Response to P1 ############

# Concrete context keeps all state/types explicit => no closures captured
struct BRContextST{G,O,P,SF}
    game::G
    oracle::O
    policy1::P
    γ::Float64
    max_depth::Int
    cache::Dict{Tuple{UInt64,Int},Float64}   # (state_key, depth) -> value
    br_policy::Dict{Tuple{UInt64,Int},Int}   # (state_key, depth) -> best_j
    statefeat::SF                             # (s, game) -> Vector{Float32}
end

state_key(ctx::BRContextST, s) = hash(ctx.statefeat(s, ctx.game))

function oracle_val(ctx::BRContextST, s)
    z = ctx.statefeat(s, ctx.game)
    return only(AZ.value(ctx.oracle, z))
end

function V(ctx::BRContextST, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return 0.0, 0.0
    elseif d ≥ ctx.max_depth
        v̂ = oracle_val(ctx, s)
        return v̂, -v̂
    end

    key = (state_key(ctx, s), d)

    # cache lookup
    if cache # FIXME: Broken - really don't want to cache
        if (v = get(ctx.cache, key, NaN); !isnan(v))
            return v
        end
    end

    A1, A2 = actions(ctx.game)
    π1,π2 = ctx.policy1(ctx.game, s)

    n,m = length(A1), length(A2)
    vals1 = Vector{Float64}(undef, n)
    vals2 = Vector{Float64}(undef, m)

    @inbounds for j in 1:m
        a2 = A2[j]
        q = 0.0
        for (i, a1) in enumerate(A1)
            sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
            r1 = AZ.zs_reward_scalar(r)
            q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
        end
        vals2[j] = q
    end
    
    @inbounds for j in 1:n
        a1 = A1[j]
        q = 0.0
        for (i, a1) in enumerate(A1)
            sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
            r1 = AZ.zs_reward_scalar(r)
            q += π1[i] * (r1 + ctx.γ * V(ctx, sp, d + 1))
        end
        vals2[j] = q
    end

    best_j = argmin(vals)
    best_val = vals[best_j]

    ctx.br_policy[key] = best_j
    ctx.cache[key] = best_val
    return best_val
end

"""
    approx_br_value_st(game::MG, oracle, policy1, s;
                       max_depth::Int=5,
                       return_policy::Bool=false)

Single-threaded limited-lookahead best response (BR) to player 1's fixed policy.
Returns P1's value at the root (and optionally the chosen BR actions per (state,depth)).

- Uses expectimax vs π₁ over horizon `max_depth`
- Bootstraps with `oracle` beyond the depth/at terminal
"""
function approx_br_value_st(game::MG, oracle, policy1, s;
                            max_depth::Int=5,
                            return_policy::Bool=false)

    ctx = BRContextST(
        game,
        oracle,
        policy1,
        discount(game),
        max_depth,
        Dict{Tuple{UInt64,Int},Float64}(),
        Dict{Tuple{UInt64,Int},Int}(),
        (s_, g_) -> MarkovGames.convert_s(Vector{Float32}, s_, g_),
    )

    root_val = V(ctx, s, 0)
    return return_policy ? (root_val, ctx.br_policy) : root_val
end

function approx_br_value(game::MG, oracle, policy1, s; max_depth::Int=5, return_policy::Bool=false)
    approx_br_value_st(game, oracle, policy1, s; max_depth, return_policy)
end


## Exploit MCTS Tree

function policy1_from_tree(game, planner, tree)
    γ = discount(game)
    (;matrix_solver) = planner
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.r[idx]) # not searched
            x,y = AZ.state_policy(planner.oracle, game, s)
            return x
        else # searched
            x,y,t = solve(matrix_solver, AZ.node_matrix_game(tree, 1.0, idx, γ))
            return x
        end
    end
end

function policy2_from_tree(game, planner, tree)
    γ = discount(game)
    (;matrix_solver) = planner
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.r[idx]) # not searched
            x,y = AZ.state_policy(planner.oracle, game, s)
            return y
        else # searched
            x,y,t = solve(matrix_solver, AZ.node_matrix_game(tree, 1.0, idx, γ))
            return y
        end
    end
end

function joint_policy_from_tree(game, planner, tree)
    γ = discount(game)
    (;matrix_solver) = planner
    return function (game, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.r[idx]) # not searched
            x,y = AZ.state_policy(planner.oracle, game, s)
            return x,y
        else # searched
            x,y,t = solve(matrix_solver, AZ.node_matrix_game(tree, 1.0, idx, γ))
            return x,y
        end
    end
end

function search_eval(planner::AlphaZeroPlanner, params::MCTSParams, game::MG, s; temperature=1.0, every=10, progress=true)
    tree = AZ.Tree(game, s)
    (;tree_queries) = params
    iter    = Int[]
    brvs1   = Float64[]
    brvs2   = Float64[]
    v       = Float64[]
    γ = discount(game)
    p = Progress(tree_queries; enabled=progress)
    for i ∈ 1:tree_queries
        AZ.simulate(params, tree, game, 1; temperature)
        if iszero(mod(i, every))
            π = joint_policy_from_tree(game, planner, tree)
            brv1, brv2 = approx_br_values(game, planner.oracle, π, s; max_depth=5)
            x,y,t = AZ.solve(AZ.node_matrix_game(tree, 1, γ))
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

function AZ.MCTSParams(planner::AlphaZeroPlanner; kwargs...)
    return MCTSParams(;
        tree_queries    = planner.max_iter,
        c               = planner.c,
        max_depth       = planner.max_depth,
        max_time        = planner.max_time,
        matrix_solver   = planner.matrix_solver,
        oracle          = planner.oracle,
        kwargs...
    )
end

##

############################ LLBR: both directions (single-threaded) ############################

# Context keeps types concrete (good for inference and JET)
struct LLBRContext{G,O,P1,P2,SF}
    game::G
    oracle::O
    π1::P1                         # (game, s) -> probs over A1
    π2::P2                         # (game, s) -> probs over A2
    γ::Float64
    max_depth::Int
    cache12::Dict{Tuple{UInt64,Int},Float64}   # cache for P2 BR vs π1  (returns P1 value)
    cache21::Dict{Tuple{UInt64,Int},Float64}   # cache for P1 BR vs π2  (returns P1 value)
    statefeat::SF                             # (s, game) -> Vector{Float32}
end

@inline function _normalize_probs(p0)
    p = Vector{Float64}(p0)
    s = sum(p)
    if s > 0.0
        @. p = p / s
    else
        fill!(p, 1.0 / length(p))
    end
    return p
end

@inline state_key(ctx::LLBRContext, s) = hash(ctx.statefeat(s, ctx.game))

@inline function oracle_val_p1(ctx::LLBRContext, s)
    sv = ctx.statefeat(s, ctx.game)
    return only(value(ctx.oracle, sv))::Float64
end

# ----- P2 best-responds to π1 (return P1's value) -----
function V12(ctx::LLBRContext, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return 0.0
    elseif d ≥ ctx.max_depth
        return oracle_val_p1(ctx, s)
    end
    
    if cache
        key = (state_key(ctx, s), d)
        if (v = get(ctx.cache12, key, NaN); !isnan(v))
            return v
        end
    end

    A1, A2 = actions(ctx.game)
    π1 = ctx.π1(ctx.game, s)
    m2 = length(A2)
    vals = Vector{Float64}(undef, m2)

    @inbounds for j in 1:m2
        a2 = A2[j]
        q = 0.0
        for (i, a1) in enumerate(A1)
            sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
            r1 = AZ.zs_reward_scalar(r)   # P1 reward (zero-sum)
            q += π1[i] * (r1 + ctx.γ * V12(ctx, sp, d + 1))
        end
        vals[j] = q
    end

    best_val = minimum(vals)
    ctx.cache12[key] = best_val
    return best_val
end

# ----- P1 best-responds to π2 (return P1's value; P2's is the negative) -----
function V21(ctx::LLBRContext, s, d::Int; cache=false)
    if isterminal(ctx.game, s)
        return 0.0
    elseif d ≥ ctx.max_depth
        return oracle_val_p1(ctx, s)
    end
    if cache
        key = (state_key(ctx, s), d)
        if (v = get(ctx.cache21, key, NaN); !isnan(v))
            return v
        end
    end

    A1, A2 = actions(ctx.game)
    π2 = ctx.π2(ctx.game, s)
    m1 = length(A1)
    vals = Vector{Float64}(undef, m1)

    @inbounds for i in 1:m1
        a1 = A1[i]
        q = 0.0
        for (j, a2) in enumerate(A2)
            sp, r = @gen(:sp, :r)(ctx.game, s, (a1, a2))
            r1 = AZ.zs_reward_scalar(r)   # P1 reward (zero-sum)
            q += π2[j] * (r1 + ctx.γ * V21(ctx, sp, d + 1))
        end
        vals[i] = q
    end

    best_val = maximum(vals)
    ctx.cache21[key] = best_val
    return best_val
end

"""
    approx_br_values_st(game::MG, oracle, π1, π2, s;
                        max_depth::Int=5)

Return `(v1_br2, v2_br1)`:

- `v1_br2`: P1 value when P2 plays a limited-lookahead best response to π₁
- `v2_br1`: P2 value when P1 plays a limited-lookahead best response to π₂

Both use horizon `max_depth`; beyond the horizon (or at terminal) they bootstrap
with the oracle's P1 value (P2's value = negative, by zero-sum).
"""
function approx_br_values_st(game::MG, oracle, π1, π2, s;
                             max_depth::Int=5, cache=false)
    ctx = LLBRContext(
        game, oracle, π1, π2,
        discount(game), max_depth,
        Dict{Tuple{UInt64,Int},Float64}(),
        Dict{Tuple{UInt64,Int},Float64}(),
        (s_, g_) -> MarkovGames.convert_s(Vector{Float32}, s_, g_),
    )
    v1_br2 = V12(ctx, s, 0; cache)        # P1's utility vs BR₂(π₁)
    v2_br1 = -V21(ctx, s, 0; cache)       # P2's utility vs BR₁(π₂)  (negate P1 in zero-sum)
    return v1_br2, v2_br1
end

# Convenience: policies from your oracle (policy head)
policy1_from_oracle(oracle) = (game, s) -> first(state_policy(oracle, game, s))
policy2_from_oracle(oracle) = (game, s) -> last(state_policy(oracle, game, s))


##
############################ Single-traversal LLBR (both directions) ############################

# Context keeps types concrete (good for inference)
struct LLBRBothCtx{G,O,P1,P2,SF}
    game::G
    oracle::O
    π1::P1                       # (game, s) -> probs over A1
    π2::P2                       # (game, s) -> probs over A2
    γ::Float64
    max_depth::Int
    cache::Dict{Tuple{UInt64,Int},NTuple{2,Float64}}  # (state_key, depth) -> (v1_br2, v2_br1)
    br2_map::Dict{Tuple{UInt64,Int},Int}              # argmin a2 for v1_br2 at (s,d)
    br1_map::Dict{Tuple{UInt64,Int},Int}              # argmax a1 for v2_br1 at (s,d)
    record_policies::Bool
    statefeat::SF                                     # (s, game) -> Vector{Float32}
end

@inline state_key(ctx::LLBRBothCtx, s) = hash(ctx.statefeat(s, ctx.game))

@inline function oracle_val_p1(ctx::LLBRBothCtx, s)
    sv = ctx.statefeat(s, ctx.game)
    return only(value(ctx.oracle, sv))::Float64
end

# Single recursion that returns both values: (v1_br2, v2_br1) from P1's perspective and then negated.
function V(ctx::LLBRBothCtx, s, d::Int; cache=false)
    if isterminal(ctx.game, s) || d ≥ ctx.max_depth
        v = oracle_val_p1(ctx, s)               # bootstrap is P1's value
        return (v, v)                           # both recursions share same bootstrap
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
- `v1_br2`: Player-1 value when Player-2 plays BR to π₁ (min over a₂, expect over π₁)
- `v2_br1`: Player-1 value when Player-1 plays BR to π₂ (max over a₁, expect over π₂)

If `return_policies=true`, returns `(v1_br2, v2_br1, br2_map, br1_map)`, where the maps
store the chosen BR action index at each visited `(state_hash, depth)`.
"""
function approx_br_values_both_st(game::MG, oracle, π1, π2, s;
                                  max_depth::Int=5, return_policies::Bool=false, cache=false)
    ctx = LLBRBothCtx(
        game, oracle, π1, π2,
        discount(game), max_depth,
        Dict{Tuple{UInt64,Int},NTuple{2,Float64}}(),
        Dict{Tuple{UInt64,Int},Int}(),
        Dict{Tuple{UInt64,Int},Int}(),
        return_policies,
        (s_, g_) -> MarkovGames.convert_s(Vector{Float32}, s_, g_),
    )
    v1_br2, v2_br1 = V(ctx, s, 0; cache)
    return return_policies ? (v1_br2, v2_br1, ctx.br2_map, ctx.br1_map) : (v1_br2, v2_br1)
end
