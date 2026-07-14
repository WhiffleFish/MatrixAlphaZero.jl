struct LossScaledTransfer
    regret_scale         :: Float64
    strategy_scale       :: Float64
    reach_power          :: Float64
    confidence_ema_decay :: Float64
    loss_tail_fraction   :: Float64
end

function LossScaledTransfer(;
        regret_scale::Real=0.25,
        strategy_scale::Real=1.0,
        reach_power::Real=1.0,
        confidence_ema_decay::Real=0.8,
        loss_tail_fraction::Real=0.2,
    )
    regret_scale >= 0 || throw(ArgumentError("regret_scale must be nonnegative"))
    strategy_scale >= 0 || throw(ArgumentError("strategy_scale must be nonnegative"))
    reach_power >= 0 || throw(ArgumentError("reach_power must be nonnegative"))
    0 <= confidence_ema_decay < 1 || throw(ArgumentError("confidence_ema_decay must be in [0, 1)"))
    0 < loss_tail_fraction <= 1 || throw(ArgumentError("loss_tail_fraction must be in (0, 1]"))
    return LossScaledTransfer(
        Float64(regret_scale),
        Float64(strategy_scale),
        Float64(reach_power),
        Float64(confidence_ema_decay),
        Float64(loss_tail_fraction),
    )
end

@kwdef struct SMOOSSearch{E, Oracle}
    oos_iterations  :: Int      = 150
    τ               :: Float64  = 0.0
    transfer_weight :: Float64  = 0.01
    # Payoff bound Δ from the regret-transfer theorem. Transferred regrets are
    # projected so their positive-part norm satisfies Φ(wR̂) ≤ wT₁|A|Δ²; Inf
    # disables the projection.
    transfer_payoff_bound :: Float64 = Inf
    loss_scaled_transfer  :: Union{Nothing, LossScaledTransfer} = nothing
    regret_confidence     :: Float64 = 0.0
    strategy_confidence   :: Float64 = 0.0
    max_depth       :: Int      = 5
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    oracle          :: Oracle
end

function with_oracle(search::SMOOSSearch, oracle; kwargs...)
    return SMOOSSearch(;
        oos_iterations = search.oos_iterations,
        τ = search.τ,
        transfer_weight = search.transfer_weight,
        transfer_payoff_bound = search.transfer_payoff_bound,
        loss_scaled_transfer = search.loss_scaled_transfer,
        regret_confidence = search.regret_confidence,
        strategy_confidence = search.strategy_confidence,
        max_depth = search.max_depth,
        ϵ = search.ϵ,
        oracle,
        kwargs...,
    )
end

abstract type RegretMatchingMethod end

struct Vanilla <: RegretMatchingMethod end
struct Plus <: RegretMatchingMethod end

struct RegretMatchingSearch{M<:RegretMatchingMethod}
    backup::Symbol
    method::M
end

function RegretMatchingSearch(; backup::Symbol=:sample, method::RegretMatchingMethod=Vanilla())
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    return RegretMatchingSearch(backup, method)
end

RegretMatchingSearch(backup::Symbol) = RegretMatchingSearch(; backup)

@kwdef struct MCTSSearch{E, Oracle}
    tree_queries    :: Int      = 150
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    max_time        :: Float64  = Inf
    search_style    :: RegretMatchingSearch = RegretMatchingSearch()
    oracle          :: Oracle
    value_target    :: Symbol   = :search
    # Regret transfer (RegretMatchingSearch + FittedRegretModel oracle only):
    # warm-start node cumulative regrets from the learned regret prior. Because
    # the regret-matching backup is exact (full matrix q = r + γV̂, no importance
    # sampling), transferred regrets are bounded per iteration by construction.
    τ                     :: Float64 = 0.0
    transfer_weight       :: Float64 = 0.0
    transfer_payoff_bound :: Float64 = Inf
    loss_scaled_transfer  :: Union{Nothing, LossScaledTransfer} = nothing
    regret_confidence     :: Float64 = 0.0
    strategy_confidence   :: Float64 = 0.0
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
        τ = search.τ,
        transfer_weight = search.transfer_weight,
        transfer_payoff_bound = search.transfer_payoff_bound,
        loss_scaled_transfer = search.loss_scaled_transfer,
        regret_confidence = search.regret_confidence,
        strategy_confidence = search.strategy_confidence,
        kwargs...,
    )
end

uniform(n::Int) = fill(inv(n), n)

zs_reward_scalar(x::Number) = x
zs_reward_scalar(x::Union{Tuple, AbstractArray}) = first(x)

function Tree end
function fitted_smoos_info end
function fitted_smoos end
function smoos_trajectory! end
function search_info end
function search end
function simulate end
function tree_policy end
function node_value end
