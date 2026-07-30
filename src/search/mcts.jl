struct SearchTree{S}
    s           :: Vector{S}
    s_children  :: Vector{Matrix{Int}}
    n_sa        :: Vector{Matrix{Float64}}
    n_s         :: Vector{Float64}
    prior       :: NTuple{2, Vector{Vector{Float32}}}
    v           :: Vector{Matrix{Float64}}
    r           :: Vector{Matrix{Float64}}
    return_sum  :: Vector{Float64}
    regret      :: NTuple{2, Vector{Vector{Float64}}}
    fresh_regret:: NTuple{2, Vector{Vector{Float64}}}
    policy_sum  :: NTuple{2, Vector{Vector{Float64}}}
    # Transferred prior held per node, kept out of the live accumulators so the
    # non-`:warmstart` modes can scale it at read time. `regret_init` is the
    # nonnegative fitted average regret R̄̂(h,·), `strategy_init` the normalized
    # fitted average strategy σ̄̂(h,·), and `prior_mass` the node's effective
    # transferred mass m_R(h) = prior_scale·q_π̄(h)^prior_reach_power.
    regret_init  :: NTuple{2, Vector{Vector{Float64}}}
    strategy_init:: NTuple{2, Vector{Vector{Float64}}}
    prior_mass  :: Vector{Float64}
end

const NO_CHILDREN = Matrix{Int}(undef, 0, 0)
const NO_COUNTS = Matrix{Float64}(undef, 0, 0)
const NO_PRIOR = Vector{Float32}(undef, 0)
const NO_FLOAT = Vector{Float64}(undef, 0)

function SearchTree(game::MG, s=rand(initialstate(game)))
    return SearchTree(
        [s],
        Matrix{Int}[NO_CHILDREN],
        Matrix{Float64}[NO_COUNTS],
        [0.0],
        ([NO_PRIOR], [NO_PRIOR]),
        [Matrix{Float64}(undef, 0, 0)],
        [Matrix{Float64}(undef, 0, 0)],
        [0.0],
        ([NO_FLOAT], [NO_FLOAT]),
        ([NO_FLOAT], [NO_FLOAT]),
        ([NO_FLOAT], [NO_FLOAT]),
        ([NO_FLOAT], [NO_FLOAT]),
        ([NO_FLOAT], [NO_FLOAT]),
        [0.0],
    )
end

Tree(::MCTSSearch, game::MG, s=rand(initialstate(game))) = SearchTree(game, s)

is_leaf(tree::SearchTree, s_idx::Int) = isempty(tree.s_children[s_idx])

function reset_search_node!(tree::SearchTree, s_idx::Int, na1::Int, na2::Int)
    tree.return_sum[s_idx] = 0.0
    tree.regret[1][s_idx] = zeros(Float64, na1)
    tree.regret[2][s_idx] = zeros(Float64, na2)
    tree.fresh_regret[1][s_idx] = zeros(Float64, na1)
    tree.fresh_regret[2][s_idx] = zeros(Float64, na2)
    tree.policy_sum[1][s_idx] = zeros(Float64, na1)
    tree.policy_sum[2][s_idx] = zeros(Float64, na2)
    tree.regret_init[1][s_idx] = zeros(Float64, na1)
    tree.regret_init[2][s_idx] = zeros(Float64, na2)
    tree.strategy_init[1][s_idx] = fill(inv(na1), na1)
    tree.strategy_init[2][s_idx] = fill(inv(na2), na2)
    tree.prior_mass[s_idx] = 0.0
    return nothing
end

function append_search_frontier!(tree::SearchTree, n_frontier::Int)
    append!(tree.return_sum, fill(0.0, n_frontier))
    foreach(tree.regret) do regret_i
        append!(regret_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.fresh_regret) do regret_i
        append!(regret_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.policy_sum) do policy_i
        append!(policy_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.regret_init) do regret_i
        append!(regret_i, fill(NO_FLOAT, n_frontier))
    end
    foreach(tree.strategy_init) do strategy_i
        append!(strategy_i, fill(NO_FLOAT, n_frontier))
    end
    append!(tree.prior_mass, fill(0.0, n_frontier))
    return nothing
end

function expand_s!(tree::SearchTree, s_idx::Int, game::MG, oracle)
    if is_leaf(tree, s_idx)
        _expand_s!(tree, s_idx, game, oracle)
    end
end

function _expand_s!(tree::SearchTree, s_idx::Int, game::MG, oracle)
    s = tree.s[s_idx]
    A1, A2 = actions(game)
    na1, na2 = length(A1), length(A2)
    s_children = zeros(Int, na1, na2)
    r = zeros(Float64, na1, na2)
    v = zeros(Float64, na1, na2)
    counter = length(tree.s) + 1
    frontier = statetype(game)[]
    nonterminal = trues(na1 * na2)

    flat_idx = 1
    for (j, a2) ∈ enumerate(A2), (i, a1) ∈ enumerate(A1)
        s_children[i, j] = counter
        sp, r_i = @gen(:sp, :r)(game, s, (a1, a2))
        push!(frontier, sp)
        r[i, j] = zs_reward_scalar(r_i)
        if isterminal(game, sp)
            nonterminal[flat_idx] = false
        end
        counter += 1
        flat_idx += 1
    end
    n_frontier = length(frontier)

    v̂ = batch_state_value(oracle, game, frontier)
    for i ∈ eachindex(v̂, nonterminal)
        v[i] = v̂[i] * nonterminal[i]
    end

    prior = state_policy(oracle, game, s)
    foreach(tree.prior, prior) do tree_prior, prior_i
        tree_prior[s_idx] = prior_i
    end

    tree.s_children[s_idx] = s_children
    tree.n_sa[s_idx] = zeros(Float64, na1, na2)
    tree.n_s[s_idx] = 0.0
    tree.v[s_idx] = v
    tree.r[s_idx] = r
    reset_search_node!(tree, s_idx, na1, na2)

    append!(tree.s, frontier)
    append!(tree.s_children, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_sa, fill(NO_COUNTS, n_frontier))
    append!(tree.n_s, fill(0.0, n_frontier))
    append!(tree.v, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.r, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    foreach(tree.prior) do prior_i
        append!(prior_i, fill(NO_PRIOR, n_frontier))
    end
    append_search_frontier!(tree, n_frontier)
    return nothing
end

function oracle_policy(params::MCTSSearch, game::MG, tree::SearchTree, s_idx::Int)
    x, y = state_policy(params.oracle, game, tree.s[s_idx])
    return Float64.(x), Float64.(y)
end

function has_prior_transfer(params::MCTSSearch)
    params.prior_scale >= 0 || throw(ArgumentError("prior_scale must be nonnegative"))
    params.regret_prior_weight >= 0 ||
        throw(ArgumentError("regret_prior_weight must be nonnegative"))
    params.strategy_prior_weight >= 0 ||
        throw(ArgumentError("strategy_prior_weight must be nonnegative"))
    params.statistic_prior_weight >= 0 ||
        throw(ArgumentError("statistic_prior_weight must be nonnegative"))
    params.prior_reach_power >= 0 ||
        throw(ArgumentError("prior_reach_power must be nonnegative"))
    component_weight = params.regret_prior_weight +
        params.strategy_prior_weight +
        params.statistic_prior_weight
    return params.prior_scale > 0 && component_weight > 0
end

# Inference warm start. `prior_scale` has the wT₁ semantics from the fitted
# transfer theory. Component weights permit deployment-time ablations without
# changing the fitted oracle, while `prior_reach_power=1` preserves the original
# attenuation by joint reach under the learned average policy.
function effective_prior_mass(params::MCTSSearch, learned_reach::Real)
    reach = clamp(Float64(learned_reach), 0.0, 1.0)
    return params.prior_scale * reach^params.prior_reach_power
end

function mcts_prior(params::MCTSSearch, game::MG, s, learned_reach::Real=1.0)
    r̂ = state_regret(params.oracle, game, s)
    ŝ = state_strategy(params.oracle, game, s)
    prior_mass = effective_prior_mass(params, learned_reach)
    regret_mass = prior_mass * params.regret_prior_weight
    strategy_mass = prior_mass * params.strategy_prior_weight
    r1 = regret_mass .* Float64.(r̂[1])
    r2 = regret_mass .* Float64.(r̂[2])
    prepare_transfer_regret!(params.search_style.method, r1)
    prepare_transfer_regret!(params.search_style.method, r2)
    s1 = strategy_mass .* normalized_or_uniform(Float64.(ŝ[1]))
    s2 = strategy_mass .* normalized_or_uniform(Float64.(ŝ[2]))
    return (r1, r2), (s1, s2)
end

# Unit-mass form of the prior: the nonnegative fitted average regret and the
# normalized fitted average strategy, with the node's effective mass returned
# separately so read-time modes can rescale it.
function mcts_unit_prior(params::MCTSSearch, game::MG, s)
    r̂ = state_regret(params.oracle, game, s)
    ŝ = state_strategy(params.oracle, game, s)
    r1 = Float64.(r̂[1])
    r2 = Float64.(r̂[2])
    prepare_transfer_regret!(params.search_style.method, r1)
    prepare_transfer_regret!(params.search_style.method, r2)
    s1 = normalized_or_uniform(Float64.(ŝ[1]))
    s2 = normalized_or_uniform(Float64.(ŝ[2]))
    return (r1, r2), (s1, s2)
end

prepare_transfer_regret!(::Vanilla, regret) = regret

function prepare_transfer_regret!(::Plus, regret)
    regret .= max.(regret, 0.0)
    return regret
end

const TRANSFER_MODES = (:warmstart, :tempered, :capped, :gated)

# Fraction of a node's full transferred mass that the node is allowed to feel,
# given the fresh evidence n it has accumulated.
#
# The cap enforces m_eff(h) = min(m_R(h), ρ·n_s), so the transfer-bias ratio
# m_R/(m_R + T₂) of the approximate-transfer bound is at most ρ/(1+ρ) at every
# node in the tree. It is also what restores monotonicity in `prior_scale` and
# in the reach attenuation: under `:warmstart` a node with no fresh regret plays
# the prior direction no matter how small its mass is, because regret matching
# normalizes away the scale.
function transfer_cap_fraction(params::MCTSSearch, mass::Float64, n::Float64)
    mass > 0 || return 0.0
    return min(1.0, params.transfer_cap_ratio * n / mass)
end

# Payoff range of a node's own matrix game, used as the scale Δ in the gate's
# concentration floor. Reading it off the node keeps the gate meaningful in
# domains whose reward magnitude varies by orders of magnitude across states.
function node_payoff_range(params::MCTSSearch, q::AbstractMatrix)
    isfinite(params.transfer_payoff_bound) && return params.transfer_payoff_bound
    isempty(q) && return 0.0
    lo, hi = extrema(q)
    return hi - lo
end

# Evidence gate. Ĝ(h) is the prior strategy pair's saddle gap on the node's own
# current payoff matrix, an observable surrogate for the (2δ_u + δ_R) transfer
# error that multiplies the bias term of the approximate-transfer bound. Mass is
# withdrawn linearly once that gap exceeds regret matching's own concentration
# floor κ·Δ·√|A|/√n_s at the node's current evidence level, so the prior keeps
# its mass only while its measured wrongness is indistinguishable from search
# noise.
function transfer_gate(params::MCTSSearch, tree::SearchTree, s_idx::Int, γ::Float64)
    n = tree.n_s[s_idx]
    n > 0 || return 1.0
    isempty(tree.r[s_idx]) && return 1.0
    q = node_matrix_game(tree, s_idx, γ)
    x̂ = tree.strategy_init[1][s_idx]
    ŷ = tree.strategy_init[2][s_idx]
    gap = maximum(q * ŷ) - minimum(transpose(q) * x̂)
    Δ = node_payoff_range(params, q)
    Δ > 0 || return 1.0
    floor_n = params.transfer_gate_tol * Δ * sqrt(maximum(size(q))) / sqrt(n)
    floor_n > 0 || return 1.0
    return clamp(2.0 - gap / floor_n, 0.0, 1.0)
end

# Effective transferred mass at a node under the active transfer mode.
function transfer_mass(params::MCTSSearch, tree::SearchTree, s_idx::Int, γ::Float64)
    mass = tree.prior_mass[s_idx]
    mass > 0 || return 0.0
    params.transfer_mode === :warmstart && return mass
    c = transfer_cap_fraction(params, mass, tree.n_s[s_idx])
    params.transfer_mode === :gated && (c *= transfer_gate(params, tree, s_idx, γ))
    return c * mass
end

read_time_transfer(params::MCTSSearch) =
    has_prior_transfer(params) && params.transfer_mode ∈ (:capped, :gated)

# Uniform tempering of the transferred regret.
#
# Regret matching normalizes, so the strategy a freshly expanded node plays is
# RM([R̄̂]₊) whatever the transferred mass is: neither `prior_scale` nor the reach
# attenuation can weaken the prior's grip on the nodes where it decides
# everything, and the raw shape of a regression residual is what sets how
# sharply the node commits. Tempering redistributes the transferred vector
# toward uniform *at fixed total mass*,
#
#   R₀(h,·) = m_R(h)·[ λ(h)·w(h,·) + (1-λ(h))·‖w(h,·)‖₁/|A| ],
#   w       = [R̄̂(h,·)]₊,
#   λ(h)    = q(h)^p‖w‖₁ / ( q(h)^p‖w‖₁ + temper·Δ̂(h) ),
#
# where Δ̂(h) is the node's own payoff range. The node's played strategy becomes
# the explicit mixture (1-λ)·uniform + λ·RM(w), so λ — not the mass — is the
# knob that decides how hard a fresh node commits to the prior, and it behaves:
#
#   * `temper = 0` gives λ = 1 and reproduces `:warmstart` exactly; as
#     `temper → ∞`, λ → 0 and the injection becomes a uniform vector of the same
#     small mass, which one real iteration overwhelms — so that limit is the cold
#     value-only solver rather than a node frozen at uniform. Holding the mass
#     fixed is what makes both endpoints recoverable.
#   * λ decreases with the reach q(h)^p, so the attenuation that was meant to
#     distrust deep unsupervised nodes finally reaches the played strategy.
#   * λ increases with ‖w‖₁ measured against Δ̂(h). This is the calibration that
#     matters: R̄̂ estimates an *average* regret, which vanishes as the source
#     solve converges, so a fitted magnitude that is small next to the local
#     payoff range means the direction is mostly fitting error and the node
#     should stay near uniform. Dividing by Δ̂(h) keeps this scale-free across
#     states whose reward magnitudes differ by orders of magnitude.
#
# Total mass is preserved, so ‖R₀‖₁ is exactly the mass the transfer theorem
# prescribes, and Φ(R₀) ≤ Φ(m_R·w) because moving mass toward the mean cannot
# raise a sum of squares: whenever the untempered warm start satisfied the
# theorem's weight condition Φ(R̊) ≤ wT₁|A|Δ², the tempered one satisfies it too.
#
# Tempering is added to the accumulator rather than mixed in at read time, which
# is what makes it behave better than `:capped`/`:gated` under RM+. In the
# accumulator, `accumulate_regret!(::Plus, …)` clips at zero, so the first update
# that contradicts the injected prior destroys it. Held outside the accumulator
# and re-added at every read, the same prior is permanent: the live regret is
# clipped at zero and therefore cannot build the negative counterweight that
# would cancel a re-injected term. Plain RM has no clipping and can build that
# counterweight, which is why read-time mixing looks sound under `Vanilla`.
#
# The construction assumes the nonnegative prior that `Plus` produces, since
# ‖w‖₁ is read off as `sum(w)`. Under `Vanilla` a transferred vector with
# nonpositive total is left untempered.
function temper_transfer_regret!(
        params::MCTSSearch,
        tree::SearchTree,
        s_idx::Int,
        game::MG,
        r1::AbstractVector,
        r2::AbstractVector,
        learned_reach::Real,
    )
    params.transfer_temper > 0 || return nothing
    q = node_matrix_game(tree, s_idx, discount(game))
    isempty(q) && return nothing
    lo, hi = extrema(q)
    Δ̂ = hi - lo
    Δ̂ > 0 || return nothing
    # `r` already carries m_R = prior_scale*regret_prior_weight*q(h)^p, so
    # dividing it out leaves q(h)^p‖w‖₁ and makes λ invariant to the mass knobs.
    unit = params.prior_scale * params.regret_prior_weight
    unit > 0 || return nothing
    for r in (r1, r2)
        mass = sum(r)
        mass > 0 || continue
        signal = mass / unit
        λ = signal / (signal + params.transfer_temper * Δ̂)
        r .= λ .* r .+ (1 - λ) * mass / length(r)
    end
    return nothing
end

function warmstart_node!(
        params::MCTSSearch,
        tree::SearchTree,
        s_idx::Int,
        game::MG;
        learned_reach::Real=1.0,
        value_prior=nothing,
        depth::Int=0,
    )
    has_prior_transfer(params) || return nothing
    depth <= params.transfer_max_depth || return nothing
    params.transfer_mode ∈ TRANSFER_MODES || throw(ArgumentError(
        "Unsupported transfer_mode=$(params.transfer_mode). Use one of $(TRANSFER_MODES).",
    ))
    prior_mass = effective_prior_mass(params, learned_reach)
    tree.prior_mass[s_idx] = prior_mass
    # One oracle query per head; both the unit-mass and the scaled forms are
    # derived from it.
    (ur1, ur2), (π1, π2) = mcts_unit_prior(params, game, tree.s[s_idx])
    if params.transfer_mode ∈ (:capped, :gated)
        # Live accumulators stay clean; the prior is mixed in at read time.
        tree.regret_init[1][s_idx] .= ur1
        tree.regret_init[2][s_idx] .= ur2
        tree.strategy_init[1][s_idx] .= π1
        tree.strategy_init[2][s_idx] .= π2
        return nothing
    end
    regret_mass = prior_mass * params.regret_prior_weight
    strategy_mass = prior_mass * params.strategy_prior_weight
    r1 = regret_mass .* ur1
    r2 = regret_mass .* ur2
    if params.transfer_mode === :tempered
        temper_transfer_regret!(params, tree, s_idx, game, r1, r2, learned_reach)
    end
    statistic_mass = prior_mass * params.statistic_prior_weight
    tree.regret[1][s_idx] .= r1
    tree.regret[2][s_idx] .= r2
    tree.policy_sum[1][s_idx] .= strategy_mass .* π1
    tree.policy_sum[2][s_idx] .= strategy_mass .* π2
    tree.n_s[s_idx] = statistic_mass
    tree.n_sa[s_idx] .= statistic_mass .* (π1 * transpose(π2))
    value_prior = isnothing(value_prior) ?
        oracle_state_value(params.oracle, game, tree.s[s_idx]) : Float64(value_prior)
    tree.return_sum[s_idx] = statistic_mass * value_prior
    return nothing
end

function mcts_root_targets(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int)
    has_prior_transfer(params) && throw(ArgumentError(
        "regret/strategy targets require prior_scale == 0; fitted priors are inference-only",
    ))
    local_iterations = sum(tree.n_sa[s_idx])
    # Fit average regret so the inference prior has the paper's direct
    # cumulative initialization: prior_scale * R_bar. The first query expands
    # a node without an RM update, so n_s is not the iteration count here.
    regret_denom = max(Float64(local_iterations), 1.0)
    yr = (
        Float64.(tree.fresh_regret[1][s_idx]) ./ regret_denom,
        Float64.(tree.fresh_regret[2][s_idx]) ./ regret_denom,
    )
    ys = (
        normalized_or_uniform(Float64.(tree.policy_sum[1][s_idx])),
        normalized_or_uniform(Float64.(tree.policy_sum[2][s_idx])),
    )
    return yr, ys
end

function empirical_policy(tree::SearchTree, s_idx::Int)
    counts = tree.n_sa[s_idx]
    x = vec(sum(counts; dims=2))
    y = vec(sum(counts; dims=1))
    return normalize_or_uniform!(Float64.(x)), normalize_or_uniform!(Float64.(y))
end

function node_matrix_game(tree::SearchTree, s_idx::Int, γ::Float64)
    return tree.r[s_idx] .+ γ .* tree.v[s_idx]
end

node_return_sum(tree::SearchTree, s_idx::Int) = tree.return_sum[s_idx]

function add_return_sum!(tree::SearchTree, s_idx::Int, value::Float64)
    tree.return_sum[s_idx] += value
    return tree.return_sum[s_idx]
end

function current_policy(::RegretMatchingSearch, tree::SearchTree, s_idx::Int)
    x = regret_matching_policy(tree.regret[1][s_idx])
    y = regret_matching_policy(tree.regret[2][s_idx])
    return x, y
end

# Regret-matching policy under read-time transfer: the node's own accumulated
# regret plus the gated/capped share of the transferred average regret.
function current_policy(
        style::RegretMatchingSearch,
        params::MCTSSearch,
        tree::SearchTree,
        s_idx::Int,
        γ::Float64,
    )
    read_time_transfer(params) || return current_policy(style, tree, s_idx)
    m = transfer_mass(params, tree, s_idx, γ) * params.regret_prior_weight
    iszero(m) && return current_policy(style, tree, s_idx)
    x = regret_matching_policy(
        tree.regret[1][s_idx] .+ m .* tree.regret_init[1][s_idx],
    )
    y = regret_matching_policy(
        tree.regret[2][s_idx] .+ m .* tree.regret_init[2][s_idx],
    )
    return x, y
end

function selection_policy(style::RegretMatchingSearch, tree::SearchTree, s_idx::Int; ϵ=0.30)
    x, y = current_policy(style, tree, s_idx)
    return eps_exploration(x, ϵ), eps_exploration(y, ϵ)
end

function accumulate_regret!(::Vanilla, regret, delta)
    regret .+= delta
    return regret
end

function accumulate_regret!(::Plus, regret, delta)
    regret .= max.(regret .+ delta, 0.0)
    return regret
end

function update_node!(style::RegretMatchingSearch, tree::SearchTree, s_idx::Int, a::CartesianIndex{2}, total::Float64, π1, π2, γ::Float64)
    q = node_matrix_game(tree, s_idx, γ)
    Δ1, Δ2 = regret_increments(style, q, a, total, π1, π2)
    accumulate_regret!(style.method, tree.regret[1][s_idx], Δ1)
    accumulate_regret!(style.method, tree.fresh_regret[1][s_idx], Δ1)
    accumulate_regret!(style.method, tree.regret[2][s_idx], Δ2)
    accumulate_regret!(style.method, tree.fresh_regret[2][s_idx], Δ2)
    tree.policy_sum[1][s_idx] .+= π1
    tree.policy_sum[2][s_idx] .+= π2
    return nothing
end

# Instantaneous counterfactual regret at a node, in two variants.
#
# `:sampled` is SM-MCTS-A's estimator: read the column the opponent actually
# played and compare it against the sampled return. It is a one-sample estimate,
# and at a 100-query depth-5 budget roughly three quarters of expanded nodes never
# exceed two visits, so most nodes never average that sample down. It also mixes
# baselines — the off-diagonal entries are compared against the sampled `total`
# while the played action is forced to zero, and `total` uses the freshly returned
# child value where `q` uses the child's running mean.
#
# `:expected` takes the exact expectation under the node's own current strategy
# pair,
#
#   Δ₁ = q σ₂ - σ₁ᵀ q σ₂,     Δ₂ = σ₁ᵀ q σ₂ - qᵀ σ₁,
#
# which is ordinary regret matching on the node's estimated matrix game. It costs
# one matrix-vector product per player, needs no extra oracle call, and uses
# strictly the same information: the sampled variant already reads a whole column
# of `q`, including entries no simulation has refined.
#
# The trade is explicit. `:expected` removes the opponent-sampling variance and
# the baseline inconsistency, and makes the node's regret independent of the
# ε-exploration, which then only decides where child values get refined. In
# exchange it leans on every entry of `q`, so unrefined entries carry more weight
# — variance for value-model bias.
function regret_increments(style::RegretMatchingSearch, q, a::CartesianIndex{2}, total::Float64, σ1, σ2)
    if style.update === :expected
        qσ2 = q * σ2
        u = dot(σ1, qσ2)
        return qσ2 .- u, u .- (transpose(q) * σ1)
    end
    i, j = Tuple(a)
    Δ1 = view(q, :, j) .- total
    Δ1[i] = 0.0
    Δ2 = total .- vec(view(q, i, :))
    Δ2[j] = 0.0
    return Δ1, Δ2
end

function zero_query_search(oracle, game::MG, s)
    x, y = state_policy(oracle, game, s)
    return Float64.(x), Float64.(y), oracle_state_value(oracle, game, s)
end

function search_info(params::MCTSSearch, game::MG, s; ϵ=0.30)
    tree = Tree(params, game, s)
    x, y, v = if isterminal(game, s)
        n1, n2 = length.(actions(game))
        uniform(n1), uniform(n2), 0.0
    elseif iszero(params.max_depth) || iszero(params.tree_queries) || iszero(params.max_time)
        zero_query_search(params.oracle, game, s)
    else
        start = time()
        for _ ∈ 1:params.tree_queries
            time() - start <= params.max_time || break
            simulate(params, tree, game, 1; ϵ)
        end
        search_result(params, tree, game, 1; ϵ)
    end
    return (x, y, v), (; tree)
end

search(params::MCTSSearch, game::MG, s; ϵ=0.30) =
    first(search_info(params, game, s; ϵ))

simulate(params::MCTSSearch, tree::SearchTree, game::MG, s_idx; ϵ=0.30, learned_reach::Float64=1.0) =
    simulate(params.search_style, params, tree, game, s_idx, 0; ϵ, learned_reach)

simulate(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30, learned_reach::Float64=1.0) =
    simulate_regret_matching(style, params, tree, game, s_idx, depth; ϵ, learned_reach)

function simulate_regret_matching(style::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, depth::Int; ϵ=0.30, learned_reach::Float64=1.0)
    s = tree.s[s_idx]
    if isterminal(game, s)
        return 0.0
    elseif depth >= params.max_depth
        leaf_value = oracle_state_value(params.oracle, game, s)
        add_return_sum!(tree, s_idx, leaf_value)
        tree.n_s[s_idx] += 1
        return leaf_value
    elseif is_leaf(tree, s_idx)
        expand_s!(tree, s_idx, game, params.oracle)
        leaf_value = oracle_state_value(params.oracle, game, s)
        warmstart_node!(params, tree, s_idx, game; learned_reach, value_prior=leaf_value, depth)
        add_return_sum!(tree, s_idx, leaf_value)
        tree.n_s[s_idx] += 1
        return leaf_value
    else
        γ = discount(game)
        # Traverse with an exploratory behavior policy, but accumulate the
        # unperturbed regret-matching policy. Self-play adds epsilon exactly
        # once when it samples the returned average strategy.
        σ1, σ2 = current_policy(style, params, tree, s_idx, γ)
        μ1, μ2 = eps_exploration(σ1, ϵ), eps_exploration(σ2, ϵ)
        a = action_idx_from_probs(μ1, μ2)
        sp_idx = tree.s_children[s_idx][a]
        i, j = Tuple(a)
        child_learned_reach = learned_reach * tree.prior[1][s_idx][i] * tree.prior[2][s_idx][j]
        vp = simulate(style, params, tree, game, sp_idx, depth + 1; ϵ, learned_reach=child_learned_reach)
        total = tree.r[s_idx][a] + γ * vp

        v̂ = tree.v[s_idx][a]
        nsa = tree.n_sa[s_idx][a]
        tree.v[s_idx][a] = v̂ + (vp - v̂) / (nsa + 1)
        tree.n_s[s_idx] += 1
        tree.n_sa[s_idx][a] += 1
        add_return_sum!(tree, s_idx, total)

        update_node!(style, tree, s_idx, a, total, σ1, σ2, γ)
        return backup_value(style, tree, s_idx, total)
    end
end

function search_result(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30)
    x, y = tree_policy(params, tree, game, s_idx; ϵ)
    v = node_value(params, tree, game, s_idx, x, y)
    return x, y, v
end

tree_policy(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30) =
    tree_policy(params.search_style, params, tree, game, s_idx; ϵ)

function tree_policy(::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int; ϵ=0.30)
    if iszero(tree.n_s[s_idx]) || isempty(tree.r[s_idx])
        return oracle_policy(params, game, tree, s_idx)
    end
    x = copy(tree.policy_sum[1][s_idx])
    y = copy(tree.policy_sum[2][s_idx])
    if read_time_transfer(params)
        # Report the weighted average of the approximate-transfer lemma:
        # (m σ̄̂ + Σₜ σₜ)/(m + T₂), with the same gated mass m the node's regret
        # matching used. Under `:warmstart` the prior enters the regret
        # accumulator with mass m but the emitted average divides by T₂ alone,
        # so the deployed strategy is not the object the lemma bounds.
        m = transfer_mass(params, tree, s_idx, discount(game)) *
            params.strategy_prior_weight
        if !iszero(m)
            x .+= m .* tree.strategy_init[1][s_idx]
            y .+= m .* tree.strategy_init[2][s_idx]
        end
    end
    return normalize_or_uniform!(x), normalize_or_uniform!(y)
end

node_value(params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, x, y) =
    node_value(params.search_style, params, tree, game, s_idx, x, y)

function node_value(::RegretMatchingSearch, params::MCTSSearch, tree::SearchTree, game::MG, s_idx::Int, x, y)
    if iszero(tree.n_s[s_idx])
        return oracle_state_value(params.oracle, game, tree.s[s_idx])
    end
    return node_return_sum(tree, s_idx) / tree.n_s[s_idx]
end

function backup_value(style::RegretMatchingSearch, tree::SearchTree, s_idx::Int, sample_value::Float64)
    return style.backup == :mean ? node_return_sum(tree, s_idx) / tree.n_s[s_idx] : sample_value
end

# Self-play for RegretMatchingSearch with a FittedRegretModel oracle. Training
# uses prior_scale=0, so these are ordinary local RM regret/strategy targets.
function mcts_regret_sim(
        params::MCTSSearch,
        game::MG,
        s;
        progress=false,
        ϵ=0.30,
        search_ϵ=ϵ,
        action_ϵ=ϵ,
        sim_depth::Int=params.max_depth,
        gae_lambda=0.95,
    )
    sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    use_search_targets = params.value_target == :search
    if !use_search_targets && params.value_target != :gae
        throw(ArgumentError("Unsupported value_target=$(params.value_target) for fitted-regret MCTS self-play. Use :search or :gae."))
    end
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    rewards = Float64[]
    values = Float64[]
    search_time_hist = Float64[]
    s_hist = Vector{Float32}[]
    regret_hist = (Vector{Float64}[], Vector{Float64}[])
    strategy_hist = (Vector{Float64}[], Vector{Float64}[])
    p = Progress(sim_depth, enabled=progress)

    while (t <= sim_depth) && !isterminal(game, s)
        search_start = time()
        (_x, _y, gv), info = search_info(params, game, s; ϵ=search_ϵ)
        yr, ys = mcts_root_targets(params, info.tree, game, 1)
        search_time = time() - search_start

        x = eps_exploration(normalized_or_uniform(ys[1]), action_ϵ)
        y = eps_exploration(normalized_or_uniform(ys[2]), action_ϵ)
        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(values, use_search_targets ? gv : oracle_state_value(params.oracle, game, s))
        push!(rewards, Float64(r))
        push!(regret_hist[1], Float64.(yr[1]))
        push!(regret_hist[2], Float64.(yr[2]))
        push!(strategy_hist[1], Float64.(ys[1]))
        push!(strategy_hist[2], Float64.(ys[2]))
        t += 1
        s = sp
        next!(p)
    end
    v_hist = if use_search_targets
        values
    else
        bootstrap = isterminal(game, s) ? 0.0 : oracle_state_value(params.oracle, game, s)
        lambda_gae_targets(rewards, values, bootstrap, γ, gae_lambda)
    end
    finish!(p)
    return (;
        s = s_hist,
        r = rewards,
        v = v_hist,
        search_time = search_time_hist,
        regret = regret_hist,
        strategy = strategy_hist,
    )
end

function mcts_sim(params::MCTSSearch, game::MG, s; progress=false, ϵ=0.30, sim_depth::Int=params.max_depth)
    sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    rewards = Float64[]
    v_hist = Float64[]
    search_time_hist = Float64[]
    s_hist = Vector{Float32}[]
    policy_hist = (Vector{Float64}[], Vector{Float64}[])
    use_search_targets = params.value_target == :search
    if !use_search_targets && params.value_target != :rollout
        throw(ArgumentError("Unsupported value_target=$(params.value_target). Use :search or :rollout."))
    end
    p = Progress(sim_depth, enabled=progress)

    while (t <= sim_depth) && !isterminal(game, s)
        search_start = time()
        x, y, gv = search(params, game, s; ϵ)
        search_time = time() - search_start

        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(rewards, r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(v_hist, use_search_targets ? gv : 0.0)
        push!(policy_hist[1], x)
        push!(policy_hist[2], y)
        if !use_search_targets
            for _t ∈ eachindex(v_hist)
                v_hist[_t] += r * γ^(t - _t)
            end
        end
        t += 1
        s = sp
        next!(p)
    end
    if !use_search_targets && !isterminal(game, s)
        vp = oracle_state_value(params.oracle, game, s)
        for _t ∈ eachindex(v_hist)
            v_hist[_t] += vp * γ^(t - _t)
        end
    end
    finish!(p)
    return (;
        s = s_hist,
        r = rewards,
        v = v_hist,
        search_time = search_time_hist,
        policy = policy_hist,
    )
end
