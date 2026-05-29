@kwdef struct SMOOSParams{E, Oracle}
    oos_iterations  :: Int      = 150
    transfer_steps  :: Int      = 0
    transfer_weight :: Float64  = 1.0
    max_depth       :: Int      = 50
    ϵ               :: E        = t -> 0.3 * (0.90 ^ (t-1))
    oracle          :: Oracle
end

function with_oracle(params::SMOOSParams, oracle)
    return SMOOSParams(;
        oos_iterations = params.oos_iterations,
        transfer_steps = params.transfer_steps,
        transfer_weight = params.transfer_weight,
        max_depth = params.max_depth,
        ϵ = params.ϵ,
        oracle,
    )
end

uniform(n::Int) = fill(inv(n), n)

zs_reward_scalar(x::Number) = x
zs_reward_scalar(x::Union{Tuple, AbstractArray}) = first(x)

function Tree end
function fitted_smoos_info end
function fitted_smoos end
function smoos_trajectory! end
