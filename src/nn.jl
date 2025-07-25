function state_value end
function batch_state_value end

function state_policy end
function batch_state_policy end


struct ActorCritic{S,A,C}
    shared::S
    actor::A
    critic::C
end

state_value(ac::ActorCritic, game::MG, s) = value(ac, convert_s(Vector{Float32}, s, game))

function batched_state_value(ac::ActorCritic, game::MG, sv)
    batch_s = mapreduce(hcat, sv) do s_i
        convert_s(Vector{Float32}, s_i, game)
    end
    return value(ac, batch_s)
end

state_policy(ac::ActorCritic, game::MG, s) = value(ac, convert_s(Vector{Float32}, s, game))

function batched_state_policy(ac::ActorCritic, game::MG, sv)
    batch_s = mapreduce(hcat, sv) do s_i
        convert_s(Vector{Float32}, s_i, game)
    end
    return policy(ac, batch_s)
end

ActorCritic(actor, critic) = ActorCritic(identity, actor, critic)

Flux.@layer :expand ActorCritic

function (ac::ActorCritic)(x; logits=false)
    encoded_input = ac.shared(x)
    value = ac.critic(encoded_input)
    policy = ac.actor(encoded_input; logits)
    return (; value, policy)
end

function loss(ac::ActorCritic, x, value_target, policy_target)
    encoded_input = ac.shared(x)
    value_loss = criticloss(ac.critic, encoded_input, value_target)
    policy_loss = actorloss(ac.actor, encoded_input, policy_target)
    return value_loss, policy_loss
end

function criticloss(critic, x::AbstractArray, v_target::AbstractArray)
    v = critic(x)
    return Flux.Losses.huber_loss(dropdims(v; dims=1), v_target)
end

function value(ac::ActorCritic, x)
    encoded_input = ac.shared(x)
    return ac.critic(encoded_input)
end

function policy(ac::ActorCritic, x)
    encoded_input = ac.shared(x)
    return ac.actor(encoded_input)
end

function getloss(ac::ActorCritic, input; value_target, policy_target)
    encoded_input = ac.shared(input)
    value_loss, value_mse = getloss(ac.critic, encoded_input; value_target)
    policy_loss = getloss(ac.actor, encoded_input; policy_target)
    return (; value_loss, value_mse, policy_loss)
end

struct MultiActor{T<:Tuple}
    actors::T
    MultiActor(t::T) where T<:Tuple = new{T}(t)
    MultiActor(args...)  = new{typeof(args)}(args)
end

Base.getindex(actor::MultiActor, i) = getindex(actor.actors, i)

(actor::MultiActor)(x; logits=false) = map(actor.actors) do actor
    out = actor(x)
    logits ? out : softmax(out)
end

function actorloss(actor::MultiActor, x::AbstractArray, p_target)
    p = actor(x; logits=true)
    return mapreduce(+, p, p_target) do p_i, p_target_i
        Flux.Losses.logitcrossentropy(p_i, p_target_i)
    end
end

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

transform_from_probs(critic::HLGaussCritic, probs::AbstractVector) = dot(critic.centers, probs)

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

## 

struct StaticActorCritic{A,C}
    actor::A
    critic::C
end

state_value(ac::StaticActorCritic, game, s) = ac.critic(s)

batch_state_value(ac::StaticActorCritic, game, sv) = map(sv) do s_i
    ac.critic(s_i)
end

state_policy(ac::StaticActorCritic, game, s) = ac.actor(s)

batch_state_policy(ac::StaticActorCritic, game, sv) = map(sv) do s_i
    ac.actor(s_i)
end
