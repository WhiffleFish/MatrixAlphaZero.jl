############ Multithreaded Limited-Lookahead Best Response to P1 ############

"""
    approx_br_value_mt(game::MG, oracle, policy1, s;
                       max_depth::Int=5,
                       return_policy::Bool=false,
                       parallel_depth::Int=2,
                       oracle_threadsafe::Bool=true)

Compute an approximate best-response value against player 1's fixed policy `policy1`,
letting player 2 minimize P1's expected return. Recurses up to `max_depth`;
beyond that (or on terminal), bootstraps with the oracle value.

- `parallel_depth`: parallelize the P2 action loop for depths `< parallel_depth`.
  (Good defaults: 1–3. Prevents oversubscription in deep trees.)
- `oracle_threadsafe`: set `false` if your oracle isn't thread-safe; we’ll lock around calls.

Returns:
- If `return_policy=false`: root value (Float64) from P1's perspective.
- If `return_policy=true`:  (root_value::Float64, br_policy::Dict) with P2’s chosen BR action per visited state.
"""
function approx_br_value_mt(game::MG{S}, oracle, policy1, _s::S;
                            max_depth::Int=5,
                            return_policy::Bool=false,
                            parallel_depth::Int=2,
                            oracle_threadsafe::Bool=true) where S

    γ = discount(game)

    # Thread-safe dicts (lock both reads & writes; Dict isn't concurrent-safe).
    cache_lock    = ReentrantLock()  # (key=(state_key, depth)) => Float64
    br_lock       = ReentrantLock()  # state => best_j
    oracle_lock   = ReentrantLock()

    cache = Dict{Tuple{UInt64,Int},Float64}()
    br_policy = Dict{S,Int}()

    # Normalize helper (robust to drift).
    # _normalize(p) = (s = sum(p); s > 0 ? p ./ s : fill(1/length(p), length(p)))

    # Make a hashable canonical key for state memoization
    # _state_key(s) = hash(MarkovGames.convert_s(Vector{Float32}, s, game))

    # Oracle value with optional lock if not thread-safe
    function oracle_val(s)
        z = MarkovGames.convert_s(Vector{Float32}, s, game)
        return Float64(only(AZ.value(oracle, z)))
        # if oracle_threadsafe
            # return Float64(only(AZ.value(oracle, z)))
        # else
            # lock(oracle_lock)
            # v = only(AZ.value(oracle, z))
            # unlock(oracle_lock)
            # return Float64(v)
        # end
    end

    function V(s, d)
        # Terminal?
        if isterminal(game, s)
            return 0.0
        end
        # Depth cutoff
        if d ≥ max_depth
            return oracle_val(s)
        end

        # key = (_state_key(s), d)
        # Check cache (lock read)
        # lock(cache_lock)
        # cached = get(cache, key, NaN)
        # unlock(cache_lock)
        # if !isnan(cached)
        #     return cached
        # end

        A1, A2 = actions(game)
        # π1 = _normalize( Vector{Float64}(policy1(game, s)) )
        π1 = policy1(game, s)

        # Compute q(j) = E_{a1~π1}[ r(s,a1,a2_j) + γ V(sp, d+1) ] for each P2 action.
        m = length(A2)
        vals = Vector{Float64}(undef, m)

        # Parallelize across P2’s actions for shallow depths
        if d < parallel_depth && m > 1
            @threads for j in 1:m
                a2 = A2[j]
                q = 0.0
                @inbounds for (i, a1) in enumerate(A1)
                    sp, r = @gen(:sp, :r)(game, s, (a1, a2))
                    r1 = AZ.zs_reward_scalar(r)
                    q += π1[i] * (r1 + γ * VL(sp, d + 1))
                end
                vals[j] = q
            end
        else
            @inbounds for j in 1:m
                a2 = A2[j]
                q = 0.0
                for (i, a1) in enumerate(A1)
                    sp, r = @gen(:sp, :r)(game, s, (a1, a2))
                    r1 = AZ.zs_reward_scalar(r)
                    q += π1[i] * (r1 + γ * V(sp, d + 1))
                end
                vals[j] = q
            end
        end

        # Choose minimizing P2 action
        best_j = argmin(vals)
        best_val = vals[best_j]

        # Record BR action for inspection
        # lock(br_lock)
        # br_policy[s] = best_j
        # unlock(br_lock)

        # Write cache
        # lock(cache_lock)
        # Double-check to avoid stomping if another thread computed meanwhile
        # if !haskey(cache, key)
        #     cache[key] = best_val
        # else
        #     best_val = cache[key]
        # end
        # unlock(cache_lock)

        return best_val
    end

    return V(_s, 0)
    # return root_val
    # return return_policy ? (root_val, br_policy) : root_val
end

########################### Convenience ###########################

"""
Use the oracle's policy head as P1's fixed policy for BR computation.
"""
policy1_from_oracle(oracle) = (game, s) -> first(state_policy(oracle, game, s))


############ Limited-Lookahead Best Response to P1 ############

"""
    approx_br_value(game::MG, oracle, policy1, s;
                    max_depth::Int=5, return_policy::Bool=false)

Compute an approximate best response value against player 1's fixed policy `policy1`
by letting player 2 minimize P1's expected return up to `max_depth`. Beyond that
depth (or on terminal states), fall back to the oracle's value estimate.

Returns:
- If `return_policy=false`: root value (Float64) from P1's perspective
- If `return_policy=true`:  (root_value::Float64, br_policy::Dict)

`br_policy` is a Dict mapping states (as seen during the search) to the
index of the chosen best-response action for player 2 at that state.
"""
function approx_br_value(game::MG{S}, oracle, policy1, s::S;
                         max_depth::Int=5, return_policy::Bool=false, cache=false) where S

    γ = discount(game)

    # Optional memoization over (state, depth)
    if cache
        _CACHE = Dict{Tuple{UInt64,Int}, Float64}()
    end

    # For a light-weight BR policy you can inspect (only for states actually visited)
    br_policy = Dict{S, Int}()

    # Normalize helper (robust to slight numeric drift)
    # _normalize(p) = (s = sum(p); s > 0 ? p ./ s : fill(1/length(p), length(p)))

    # Hash a state for caching (you can swap to something game-specific if you have it)
    # _key(s, d) = (hash(s), d)

    function V(s, d)
        if isterminal(game, s) # Terminal?
            return 0.0
        elseif d ≥ max_depth # Depth cutoff -> oracle bootstrap
            return AZ.state_value(oracle, game, s)
        end
        if cache
            key = _key(s, d)
            if haskey(_CACHE, key)
                return _CACHE[key]
            end
        end

        A1, A2 = actions(game)
        π1 = policy1(game, s)

        # Player 2 chooses action minimizing P1's expected value
        best_val = Inf
        best_j   = 1

        for (j, a2) in enumerate(A2)
            q = 0.0
            for (i, a1) in enumerate(A1)
                sp, r = @gen(:sp, :r)(game, s, (a1, a2))
                r1 = AZ.zs_reward_scalar(r)  # P1's reward component
                q += π1[i] * (r1 + γ * V(sp, d + 1))
            end
            if q < best_val
                best_val = q
                best_j   = j
            end
        end

        br_policy[s] = best_j
        if cache
            _CACHE[key] = best_val
        end
        return best_val
    end

    root_val = V(s, 0)
    return return_policy ? (root_val, br_policy) : root_val
end

########################### Helpers ###########################

"""
A convenience wrapper if you want to use your current oracle's policy
as P1's fixed policy for the BR computation.
"""
policy1_from_oracle(oracle) = (game, s) -> first(AZ.state_policy(oracle, game, s))
