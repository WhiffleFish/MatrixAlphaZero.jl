@kwdef struct SMOOSSearch{E, Oracle}
    oos_iterations  :: Int      = 150
    τ               :: Float64  = 0.0
    transfer_weight :: Float64  = 0.01
    max_depth       :: Int      = 5
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    oracle          :: Oracle
end

function with_oracle(search::SMOOSSearch, oracle; kwargs...)
    return SMOOSSearch(;
        oos_iterations = search.oos_iterations,
        τ = search.τ,
        transfer_weight = search.transfer_weight,
        max_depth = search.max_depth,
        ϵ = search.ϵ,
        oracle,
        kwargs...,
    )
end

struct RegretMatchingSearch
    backup::Symbol
end

function RegretMatchingSearch(; backup::Symbol=:sample)
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    return RegretMatchingSearch(backup)
end

@kwdef struct MCTSSearch{E, Oracle}
    tree_queries    :: Int      = 150
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    max_time        :: Float64  = Inf
    search_style    :: RegretMatchingSearch = RegretMatchingSearch()
    oracle          :: Oracle
    value_target    :: Symbol   = :search
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
