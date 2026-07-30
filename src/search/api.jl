abstract type RegretMatchingMethod end

struct Vanilla <: RegretMatchingMethod end
struct Plus <: RegretMatchingMethod end

struct RegretMatchingSearch{M<:RegretMatchingMethod}
    backup::Symbol
    method::M
    # How each node's instantaneous regret is formed.
    #
    #   :sampled  — SM-MCTS-A's estimator: the opponent's realized column against
    #               the sampled return. A one-sample estimate at every visit.
    #   :expected — the exact expectation under the node's own strategy pair,
    #               Δ₁ = qσ₂ - σ₁ᵀqσ₂, i.e. regret matching on the node's
    #               estimated matrix game. Same information, no sampling variance,
    #               but it weights every entry of q including unrefined ones.
    update::Symbol
end

function RegretMatchingSearch(;
        backup::Symbol=:sample,
        method::RegretMatchingMethod=Vanilla(),
        update::Symbol=:sampled,
    )
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    update ∈ (:sampled, :expected) ||
        throw(ArgumentError("Unsupported update=$(update). Use :sampled or :expected."))
    return RegretMatchingSearch(backup, method, update)
end

RegretMatchingSearch(backup::Symbol) = RegretMatchingSearch(; backup)

@kwdef struct MCTSSearch{E, Oracle}
    tree_queries    :: Int      = 150
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    max_time        :: Float64  = Inf
    search_style    :: RegretMatchingSearch = RegretMatchingSearch()
    oracle          :: Oracle
    # Value supervision mode. `:search` uses each root search value. Fitted
    # regret self-play also supports `:gae`, while policy self-play supports
    # `:rollout` for its bootstrapped environment return.
    value_target    :: Symbol   = :search
    # Inference-only tree warm start. At a node with learned joint-policy reach
    # q(h), each transferred component receives mass
    # prior_scale*q(h)^prior_reach_power times its component weight. The default
    # weights preserve the original coupled warm start. Training requires
    # prior_scale to remain zero.
    prior_scale     :: Float64  = 0.0
    regret_prior_weight    :: Float64 = 1.0
    strategy_prior_weight  :: Float64 = 1.0
    statistic_prior_weight :: Float64 = 1.0
    prior_reach_power      :: Float64 = 1.0
    # How the transferred prior enters the node solver.
    #
    #   :warmstart — add the prior into the node's regret/strategy accumulators
    #                on expansion. Because regret matching normalizes, a node
    #                whose own regret is still zero plays the prior direction
    #                regardless of `prior_scale` or the reach attenuation, so
    #                neither knob damps the prior where it acts most.
    #   :tempered  — `:warmstart` with the transferred vector redistributed
    #                toward uniform at fixed total mass, so the node plays the
    #                explicit mixture (1-λ)·uniform + λ·RM([R̄̂]₊) with
    #                λ = q(h)^p‖[R̄̂]₊‖₁ / (q(h)^p‖[R̄̂]₊‖₁ + transfer_temper·Δ̂(h))
    #                and Δ̂(h) the node's own payoff range. λ falls with the reach
    #                attenuation and rises with the fitted regret magnitude
    #                relative to the local payoff scale. `transfer_temper = 0`
    #                recovers `:warmstart`; `transfer_temper → ∞` recovers the
    #                cold value-only solver.
    #   :capped    — accumulators stay clean and the prior is mixed in at read
    #                time with effective mass min(m_R(h), transfer_cap_ratio*n_s),
    #                so injected mass never exceeds a fixed multiple of the
    #                node's own fresh evidence.
    #   :gated     — :capped, further multiplied by an evidence gate that
    #                withdraws mass once the prior strategy pair's saddle gap on
    #                the node's own payoff matrix exceeds regret matching's
    #                concentration floor at the node's current evidence level.
    transfer_mode          :: Symbol  = :warmstart
    # Uniform tempering weight for `:tempered`, in units of the node's payoff
    # range per unit of prior mass. Zero leaves the warm start untempered.
    transfer_temper        :: Float64 = 0.0
    # Deepest tree depth that receives a warm start; deeper nodes start cold.
    # The regret and average-strategy heads are supervised only at environment
    # decision states, i.e. at search roots, so every internal node is an
    # out-of-distribution query. Setting this to 0 restricts the transfer to the
    # states it was actually fitted on, which isolates how much of the measured
    # transfer effect comes from applying the prior off its training support.
    transfer_max_depth     :: Int     = typemax(Int)
    # Cap ratio ρ: effective prior mass ≤ ρ·n_s, bounding the transfer-bias
    # ratio m_R/(m_R + T₂) by ρ/(1+ρ) uniformly over the tree.
    transfer_cap_ratio     :: Float64 = 0.5
    # Gate tolerance κ: full mass while the measured prior gap is within
    # κ·Δ·√|A|/√n_s, zero mass beyond twice that.
    transfer_gate_tol      :: Float64 = 1.0
    # Payoff range used for the gate's concentration floor. `Inf` reads the
    # range off the node's own payoff matrix, which is what makes the gate
    # scale-free across states with very different reward magnitudes.
    transfer_payoff_bound  :: Float64 = Inf
end

function with_oracle(search::MCTSSearch, oracle; kwargs...)
    return MCTSSearch(;
        tree_queries = search.tree_queries,
        max_depth = search.max_depth,
        ϵ = search.ϵ,
        max_time = search.max_time,
        search_style = search.search_style,
        oracle,
        value_target = search.value_target,
        prior_scale = search.prior_scale,
        regret_prior_weight = search.regret_prior_weight,
        strategy_prior_weight = search.strategy_prior_weight,
        statistic_prior_weight = search.statistic_prior_weight,
        prior_reach_power = search.prior_reach_power,
        transfer_mode = search.transfer_mode,
        transfer_temper = search.transfer_temper,
        transfer_max_depth = search.transfer_max_depth,
        transfer_cap_ratio = search.transfer_cap_ratio,
        transfer_gate_tol = search.transfer_gate_tol,
        transfer_payoff_bound = search.transfer_payoff_bound,
        kwargs...,
    )
end

uniform(n::Int) = fill(inv(n), n)

zs_reward_scalar(x::Number) = x
zs_reward_scalar(x::Union{Tuple, AbstractArray}) = first(x)

function Tree end
function search_info end
function search end
function simulate end
function tree_policy end
function node_value end
