@kwdef struct SMOOSParams{E, Oracle}
    oos_iterations  :: Int      = 150
    τ               :: Float64  = 0.0
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    oracle          :: Oracle
end

function with_oracle(params::SMOOSParams, oracle; kwargs...)
    return SMOOSParams(;
        oos_iterations = params.oos_iterations,
        τ = params.τ,
        max_depth = params.max_depth,
        ϵ = params.ϵ,
        oracle,
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
