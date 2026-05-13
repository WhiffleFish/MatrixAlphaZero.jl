struct RegretMatchingSearch
    backup::Symbol
end

function RegretMatchingSearch(; backup::Symbol=:sample)
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    return RegretMatchingSearch(backup)
end

@kwdef struct MCTSParams{E, Oracle}
    tree_queries    :: Int      = 150
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    max_time        :: Float64  = Inf
    search_style    :: RegretMatchingSearch = RegretMatchingSearch()
    oracle          :: Oracle
    value_target    :: Symbol   = :search
end

function with_oracle(params::MCTSParams, oracle)
    return MCTSParams(;
        tree_queries = params.tree_queries,
        max_depth = params.max_depth,
        ϵ = params.ϵ,
        max_time = params.max_time,
        search_style = params.search_style,
        oracle,
        value_target = params.value_target
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
