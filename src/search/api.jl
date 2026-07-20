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
    # Value supervision mode. `:search` uses each root search value. Fitted
    # regret self-play also supports `:gae`, while policy self-play supports
    # `:rollout` for its bootstrapped environment return.
    value_target    :: Symbol   = :search
    # Inference-only regret and average-strategy warm start. At a node with
    # learned joint-policy reach q(h), both fitted priors receive cumulative
    # weight prior_scale*q(h). Training requires this to remain zero.
    prior_scale     :: Float64  = 0.0
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
