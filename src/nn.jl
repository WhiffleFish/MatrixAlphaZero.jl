function state_value end
function batch_state_value end

function state_regret end
function batch_state_regret end

function state_strategy end
function batch_state_strategy end

# Convenience aliases for downstream policy-oriented code.
state_policy(args...) = state_strategy(args...)
batch_state_policy(args...) = batch_state_strategy(args...)
policy(args...; kwargs...) = strategy(args...; kwargs...)

struct FittedRegretModel{S,R,A,C}
    shared::S
    regret_head::R
    strategy_head::A
    critic::C
end

FittedRegretModel(regret_head, strategy_head, critic) =
    FittedRegretModel(identity, regret_head, strategy_head, critic)

Flux.@layer :expand FittedRegretModel

function (model::FittedRegretModel)(x; logits=false)
    encoded_input = model.shared(x)
    value = model.critic(encoded_input)
    regret = model.regret_head(encoded_input; logits=true)
    strategy = model.strategy_head(encoded_input; logits)
    return (; value, regret, strategy)
end

function value(model::FittedRegretModel, x)
    encoded_input = model.shared(x)
    return model.critic(encoded_input)
end

function regret(model::FittedRegretModel, x)
    encoded_input = model.shared(x)
    return model.regret_head(encoded_input; logits=true)
end

function strategy(model::FittedRegretModel, x; logits=false)
    encoded_input = model.shared(x)
    return model.strategy_head(encoded_input; logits)
end

state_value(model::FittedRegretModel, game::MG, s) =
    value(model, convert_s(Vector{Float32}, s, game))

state_regret(model::FittedRegretModel, game::MG, s) =
    regret(model, convert_s(Vector{Float32}, s, game))

state_strategy(model::FittedRegretModel, game::MG, s) =
    strategy(model, convert_s(Vector{Float32}, s, game))

function batch_state_value(model::FittedRegretModel, game::MG, sv)
    batch_s = mapreduce(hcat, sv) do s_i
        convert_s(Vector{Float32}, s_i, game)
    end
    return value(model, batch_s)
end

function batch_state_regret(model::FittedRegretModel, game::MG, sv)
    batch_s = mapreduce(hcat, sv) do s_i
        convert_s(Vector{Float32}, s_i, game)
    end
    return regret(model, batch_s)
end

function batch_state_strategy(model::FittedRegretModel, game::MG, sv)
    batch_s = mapreduce(hcat, sv) do s_i
        convert_s(Vector{Float32}, s_i, game)
    end
    return strategy(model, batch_s)
end

function loss(model::FittedRegretModel, x, value_target, regret_target, strategy_target)
    encoded_input = model.shared(x)
    value_loss = criticloss(model.critic, encoded_input, value_target)
    regret_loss = fitted_regret_loss(model.regret_head, encoded_input, regret_target)
    strategy_loss = fitted_strategy_loss(model.strategy_head, encoded_input, strategy_target)
    return value_loss, regret_loss, strategy_loss
end

function getloss(model::FittedRegretModel, input; value_target, regret_target, strategy_target)
    encoded_input = model.shared(input)
    value_loss, value_mse = getloss(model.critic, encoded_input; value_target)
    regret_loss = fitted_regret_loss(model.regret_head, encoded_input, regret_target)
    strategy_loss = fitted_strategy_loss(model.strategy_head, encoded_input, strategy_target)
    return (; value_loss, value_mse, regret_loss, strategy_loss)
end

function criticloss(critic, x::AbstractArray, v_target::AbstractArray)
    v = critic(x)
    return Flux.Losses.huber_loss(dropdims(v; dims=1), v_target)
end

struct MultiActor{T<:Tuple}
    actors::T
    MultiActor(t::T) where T<:Tuple = new{T}(t)
    MultiActor(args...)  = new{typeof(args)}(args)
end

Base.getindex(actor::MultiActor, i) = getindex(actor.actors, i)

(actor::MultiActor)(x; logits=false) = map(actor.actors) do actor_i
    out = actor_i(x)
    logits ? out : softmax(out)
end

fitted_regret_loss(head::MultiActor, x::AbstractArray, r_target) =
    mapreduce(+, head(x; logits=true), r_target) do r_i, target_i
        Flux.Losses.huber_loss(r_i, target_i)
    end

function fitted_strategy_loss(head::MultiActor, x::AbstractArray, s_target)
    s = head(x; logits=true)
    return mapreduce(+, s, s_target) do s_i, target_i
        Flux.Losses.logitcrossentropy(s_i, target_i)
    end
end

actorloss(actor::MultiActor, x::AbstractArray, p_target) =
    fitted_strategy_loss(actor, x, p_target)

struct HLGaussCritic{N,S}
    net::N
    support::S
    centers::Vector{Float32}
    sigma::Float64
end

Flux.@layer :expand HLGaussCritic

Flux.trainable(critic::HLGaussCritic) = (; net=critic.net)
Flux.Functors.children(critic::HLGaussCritic) = (; net=critic.net)

function (critic::HLGaussCritic)(x; logits=false)
    out = critic.net(x)
    if logits
        return out
    else
        probs = softmax(out; dims=1)
        return transform_from_probs(critic, probs)
    end
end

function HLGaussCritic(net, min_val, max_val, n_bins::Int=64, smooth_ratio=0.75)
    support = range(min_val, max_val, length=n_bins+1)
    centers = (support[1:end-1] .+ support[2:end]) ./ 2
    sigma = step(support) * smooth_ratio
    return HLGaussCritic(net, support, convert(Vector{Float32}, centers), sigma)
end

function transform_to_probs(critic::HLGaussCritic, target::Number)
    (; sigma, support, centers) = critic
    if target > maximum(support)
        ps = zero(centers)
        ps[end] = one(eltype(centers))
        return ps
    elseif target < minimum(support)
        ps = zero(centers)
        ps[1] = one(eltype(centers))
        return ps
    else
        cdf_evals = erf.((support .- target) ./ (sqrt(2) * sigma))
        z = last(cdf_evals) - first(cdf_evals)
        return convert(Vector{Float32}, diff(cdf_evals) ./ z)
    end
end

function transform_to_probs(critic::HLGaussCritic, target::AbstractVector)
    return mapreduce(hcat, target) do t
        transform_to_probs(critic, t)
    end
end

transform_from_probs(critic::HLGaussCritic, probs::AbstractVector) =
    dot(critic.centers, probs)

function transform_from_probs(critic::HLGaussCritic, probs::AbstractMatrix)
    return map(eachcol(probs)) do p
        transform_from_probs(critic, p)
    end
end

prepare_target(critic, target::AbstractArray) = target

function prepare_target(critic::HLGaussCritic, target::AbstractArray)
    return mapreduce(hcat, target) do t
        transform_to_probs(critic, t)
    end
end

function criticloss(critic::HLGaussCritic, x::AbstractArray, y::AbstractArray)
    ŷ = critic(x; logits=true)
    return Flux.Losses.logitcrossentropy(ŷ, y)
end

struct StaticFittedRegretModel{R,A,C}
    regret::R
    strategy::A
    critic::C
end

state_value(model::StaticFittedRegretModel, game, s) = model.critic(s)

batch_state_value(model::StaticFittedRegretModel, game, sv) = map(sv) do s_i
    model.critic(s_i)
end

state_regret(model::StaticFittedRegretModel, game, s) = model.regret(s)

batch_state_regret(model::StaticFittedRegretModel, game, sv) = map(sv) do s_i
    model.regret(s_i)
end

state_strategy(model::StaticFittedRegretModel, game, s) = model.strategy(s)

batch_state_strategy(model::StaticFittedRegretModel, game, sv) = map(sv) do s_i
    model.strategy(s_i)
end
