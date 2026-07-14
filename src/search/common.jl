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

uses_loss_scaled_transfer(params) = !isnothing(params.loss_scaled_transfer)
search_budget(params::SMOOSSearch) = params.oos_iterations
search_budget(params::MCTSSearch) = params.tree_queries

function transfer_pseudo_masses(params, learned_reach::Real=1.0)
    config = params.loss_scaled_transfer
    isnothing(config) && return nothing
    reach = clamp(Float64(learned_reach), 0.0, 1.0)^config.reach_power
    budget = max(Float64(search_budget(params)), 0.0)
    source_mass = max(params.τ, 0.0)
    regret_confidence = clamp(params.regret_confidence, 0.0, 1.0)
    strategy_confidence = clamp(params.strategy_confidence, 0.0, 1.0)
    regret_mass = min(source_mass, config.regret_scale * regret_confidence * budget * reach)
    strategy_mass = min(source_mass, config.strategy_scale * strategy_confidence * budget * reach)
    return regret_mass, strategy_mass
end
