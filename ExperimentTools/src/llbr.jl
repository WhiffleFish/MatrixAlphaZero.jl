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


########################### Convenience ###########################

"""
Use the oracle's policy head as P1's fixed policy for BR computation.
"""
policy1_from_oracle(oracle) = (game, s) -> first(AZ.state_policy(oracle, game, s))


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

# Top-level recursion (no inner function => no capture)
function V(ctx::BRContextST, s, d::Int)
    if isterminal(ctx.game, s)
        return 0.0
    elseif d ≥ ctx.max_depth
        return oracle_val(ctx, s)
    end

    key = (state_key(ctx, s), d)

    # cache lookup
    if (v = get(ctx.cache, key, NaN); !isnan(v))
        return v
    end

    A1, A2 = actions(ctx.game)
    π1 = ctx.policy1(ctx.game, s)

    m = length(A2)
    vals = Vector{Float64}(undef, m)

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

approx_br_value(game::MG, oracle, policy1, s; max_depth::Int=5, return_policy::Bool=false) =
    approx_br_value_st(game, oracle, policy1, s; max_depth, return_policy)
