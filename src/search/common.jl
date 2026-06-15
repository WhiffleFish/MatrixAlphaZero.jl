function normalize_or_uniform!(x::AbstractVector)
    isempty(x) && return x
    s = sum(x)
    if isfinite(s) && s > 0
        x ./= s
    else
        x .= inv(length(x))
    end
    return x
end

normalized_or_uniform(x::AbstractVector) = normalize_or_uniform!(copy(x))

eps_exploration(p, ϵ) = inv(length(p)) .* ϵ .+ (1 .- ϵ) .* p

function match!(x, r)
    rs = zero(eltype(r))
    for i ∈ eachindex(x,r)
        r_i = r[i]
        if r_i > zero(r_i)
            rs += r_i
            x[i] = r_i
        else
            x[i] = zero(r_i)
        end
    end
    rs > zero(rs) ? x ./= rs : x .= inv(length(x))
end

function regret_matching_policy(regret::AbstractVector)
    policy = zeros(Float64, length(regret))
    return match!(policy, regret)
end

function action_idx_from_probs(x, y)
    x = normalized_or_uniform(x)
    y = normalized_or_uniform(y)
    return CartesianIndex(
        rand(Categorical(x)),
        rand(Categorical(y)),
    )
end

function oracle_state_value(oracle, game::MG, s)
    return Float64(only(value(oracle, MarkovGames.convert_s(Vector{Float32}, s, game))))
end
