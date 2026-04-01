abstract type AbstractSearchStyle end
abstract type AbstractSearchTree end
abstract type AbstractBanditTree <: AbstractSearchTree end

struct MatrixGameSearch{MS} <: AbstractSearchStyle
    c             :: Float64
    matrix_solver :: MS
end

function MatrixGameSearch(; c::Real=1.0, matrix_solver=RegretSolver(100))
    return MatrixGameSearch(Float64(c), matrix_solver)
end

struct RegretMatchingSearch <: AbstractSearchStyle
    backup::Symbol
    target_policy::Symbol
end

function RegretMatchingSearch(; backup::Symbol=:sample, target_policy::Symbol=:average)
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    target_policy ∈ (:average, :empirical, :current) || throw(ArgumentError("Unsupported target_policy=$(target_policy). Use :average, :empirical, or :current."))
    return RegretMatchingSearch(backup, target_policy)
end

struct Exp3Search <: AbstractSearchStyle
    backup::Symbol
    target_policy::Symbol
    η::Float64
end

function Exp3Search(; backup::Symbol=:sample, target_policy::Symbol=:empirical, η::Real=NaN)
    backup ∈ (:sample, :mean) || throw(ArgumentError("Unsupported backup=$(backup). Use :sample or :mean."))
    target_policy ∈ (:empirical, :current) || throw(ArgumentError("Unsupported target_policy=$(target_policy). Use :empirical or :current."))
    (isnan(η) || η > 0) || throw(ArgumentError("η must be positive when provided."))
    return Exp3Search(backup, target_policy, Float64(η))
end

@kwdef struct MCTSParams{E, Oracle, SS<:AbstractSearchStyle}
    tree_queries    :: Int      = 150
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    max_time        :: Float64  = Inf
    search_style    :: SS       = MatrixGameSearch()
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
