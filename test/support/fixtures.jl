module Fixtures

using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Random

const AZ = MatrixAlphaZero

struct ScalarMatrixGame <: MG{Bool, NTuple{2, Int}}
    rewards::Matrix{Float64}
end

ScalarMatrixGame() = ScalarMatrixGame([
    1.0 -1.0
    0.5 2.0
])

POMDPs.initialstate(::ScalarMatrixGame) = Deterministic(false)
POMDPs.discount(::ScalarMatrixGame) = 1.0
POMDPs.isterminal(::ScalarMatrixGame, s::Bool) = s
POMDPs.actions(g::ScalarMatrixGame) = (axes(g.rewards, 1), axes(g.rewards, 2))
POMDPs.gen(g::ScalarMatrixGame, s::Bool, a::NTuple{2, Int}, rng::Random.AbstractRNG=Random.default_rng()) = (sp=true, r=g.rewards[a...])
POMDPs.convert_s(::Type{Vector{Float32}}, s::Bool, ::ScalarMatrixGame) = Float32[s ? 1 : 0]

struct TwoStepGame <: MG{Int, NTuple{2, Int}}
    rewards0::Matrix{Float64}
    rewards1::Matrix{Float64}
    γ::Float64
end

TwoStepGame() = TwoStepGame(
    [
        1.0 2.0
        0.0 1.0
    ],
    [
        0.5 0.5
        0.5 0.5
    ],
    0.5
)

POMDPs.initialstate(::TwoStepGame) = Deterministic(0)
POMDPs.discount(g::TwoStepGame) = g.γ
POMDPs.isterminal(::TwoStepGame, s::Int) = s >= 2
POMDPs.actions(::TwoStepGame) = (1:2, 1:2)
POMDPs.convert_s(::Type{Vector{Float32}}, s::Int, ::TwoStepGame) = Float32[s]
function POMDPs.gen(g::TwoStepGame, s::Int, a::NTuple{2, Int}, rng::Random.AbstractRNG=Random.default_rng())
    if s == 0
        return (sp=1, r=g.rewards0[a...])
    elseif s == 1
        return (sp=2, r=g.rewards1[a...])
    else
        return (sp=2, r=0.0)
    end
end

state_key(s::Bool) = Float32(s ? 1 : 0)
state_key(s::Int) = Float32(s)
state_key(x::AbstractVector) = Float32(only(x))

function uniform_policy(game)
    A1, A2 = actions(game)
    return (
        fill(Float32(inv(length(A1))), length(A1)),
        fill(Float32(inv(length(A2))), length(A2)),
    )
end

struct TableOracle
    values::Dict{Float32, Float32}
    policies::Dict{Float32, NTuple{2, Vector{Float32}}}
end

TableOracle(; values=Dict{Float32, Float32}(), policies=Dict{Float32, NTuple{2, Vector{Float32}}}()) = TableOracle(values, policies)

function AZ.value(o::TableOracle, x::AbstractVector)
    return Float32[get(o.values, state_key(x), 0f0)]
end

function AZ.value(o::TableOracle, x::AbstractMatrix)
    vals = Float32[get(o.values, state_key(col), 0f0) for col in eachcol(x)]
    return reshape(vals, 1, :)
end

function AZ.state_policy(o::TableOracle, game::MG, s)
    return get(o.policies, state_key(s), uniform_policy(game))
end

function AZ.batch_state_policy(o::TableOracle, game::MG, sv)
    π = map(s -> AZ.state_policy(o, game, s), sv)
    return (
        map(first, π),
        map(last, π),
    )
end

function AZ.batch_state_value(o::TableOracle, game::MG, sv)
    return Float32[get(o.values, state_key(s), 0f0) for s in sv]
end

struct GreedyMatrixSolver end

function MarkovGames.solve(::GreedyMatrixSolver, A::AbstractMatrix)
    idx = argmax(A)
    x = zeros(Float64, size(A, 1))
    y = zeros(Float64, size(A, 2))
    x[idx[1]] = 1.0
    y[idx[2]] = 1.0
    return x, y, A[idx]
end

function MarkovGames.solve(::GreedyMatrixSolver, A::AbstractMatrix, B::AbstractMatrix)
    return MarkovGames.solve(GreedyMatrixSolver(), A)
end

function simple_actor_critic(; input_dim::Int=1, hidden_dim::Int=4, action_dims=(2, 2))
    shared = Dense(input_dim => hidden_dim, tanh)
    actor1 = Dense(hidden_dim => action_dims[1])
    actor2 = Dense(hidden_dim => action_dims[2])
    critic = Dense(hidden_dim => 1)

    shared.weight .= reshape(Float32.(range(-0.4, 0.4; length=length(shared.weight))), size(shared.weight))
    shared.bias .= Float32.(range(-0.2, 0.2; length=length(shared.bias)))
    actor1.weight .= reshape(Float32.(range(-0.3, 0.3; length=length(actor1.weight))), size(actor1.weight))
    actor1.bias .= Float32.(range(-0.1, 0.1; length=length(actor1.bias)))
    actor2.weight .= reshape(Float32.(range(0.25, -0.25; length=length(actor2.weight))), size(actor2.weight))
    actor2.bias .= Float32.(range(0.05, -0.05; length=length(actor2.bias)))
    critic.weight .= reshape(Float32.(range(-0.2, 0.2; length=length(critic.weight))), size(critic.weight))
    critic.bias .= Float32[0.1]

    return AZ.ActorCritic(shared, AZ.MultiActor(actor1, actor2), critic)
end

function AZ.getloss(critic::Dense, x; value_target)
    ŷ = vec(critic(x))
    return Flux.Losses.huber_loss(ŷ, value_target), Flux.Losses.mse(ŷ, value_target)
end

AZ.getloss(actor::AZ.MultiActor, x; policy_target) = AZ.actorloss(actor, x, policy_target)

end
