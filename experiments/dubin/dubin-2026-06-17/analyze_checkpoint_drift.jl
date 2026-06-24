using Pkg
Pkg.activate("experiments")

using DelimitedFiles
using Flux
using LinearAlgebra
using MarkovGames
using MatrixAlphaZero
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random
using Statistics

const AZ = MatrixAlphaZero
const EXPERIMENT_DIR = @__DIR__
const CHECKPOINTS = (0, 100, 250, 500, 700, 780, 820, 870, 950, 1050, 1130, 1221)
const ANCHOR_ITER = 820
const STATE_COUNT = 2048

function fixed_state_suite(game, n::Int; seed::Int=20260617)
    rng = MersenneTwister(seed)
    states = JointDubinState[]
    while length(states) < n
        attacker = SA[10rand(rng), 10rand(rng), 2pi * rand(rng) - pi]
        defender = SA[10rand(rng), 10rand(rng), 2pi * rand(rng) - pi]
        state = JointDubinState(attacker, defender)
        isterminal(game, state) || push!(states, state)
    end
    return states
end

flatten_parameters(model) =
    reduce(vcat, (vec(Float64.(p)) for p in Flux.trainables(model)); init=Float64[])

relative_distance(current, reference) =
    norm(current .- reference) / max(norm(reference), eps(Float64))

rms_distance(current, reference) = sqrt(mean(abs2, current .- reference))

function tuple_rms_distance(current, reference)
    return mean(rms_distance(current[i], reference[i]) for i in eachindex(current))
end

function mean_entropy(policy)
    return mean(policy) do player_policy
        total = 0.0
        for p in player_policy
            p > 0 && (total -= p * log(p))
        end
        total / size(player_policy, 2)
    end
end

function mean_kl(policy, reference)
    return mean(eachindex(policy)) do player
        total = 0.0
        n = size(policy[player], 2)
        for j in axes(policy[player], 2), i in axes(policy[player], 1)
            p = policy[player][i, j]
            p > 0 && (total += p * log(p / max(reference[player][i, j], eps(eltype(reference[player])))))
        end
        total / n
    end
end

function argmax_agreement(policy, reference)
    return mean(eachindex(policy)) do player
        matches = count(axes(policy[player], 2)) do j
            argmax(view(policy[player], :, j)) == argmax(view(reference[player], :, j))
        end
        matches / size(policy[player], 2)
    end
end

function mean_positive_regret_l2(regret)
    return mean(eachindex(regret)) do player
        mean(axes(regret[player], 2)) do j
            norm(max.(view(regret[player], :, j), 0))
        end
    end
end

checkpoint_path(iter::Int) = joinpath(
    EXPERIMENT_DIR,
    "models_smoos",
    "oracle$(AZ.iter2string(iter)).jld2",
)

function load_checkpoint!(model, iter::Int)
    Flux.loadmodel!(model, checkpoint_path(iter))
    return model
end

function outputs(model, input)
    return (
        value = vec(Float64.(AZ.value(model, input))),
        regret = map(x -> Float64.(x), AZ.regret(model, input)),
        strategy = map(x -> Float64.(x), AZ.strategy(model, input)),
    )
end

game = DubinMG(V=(1.0, 1.0))
states = fixed_state_suite(game, STATE_COUNT)
input = mapreduce(hcat, states) do state
    MarkovGames.convert_s(Vector{Float32}, state, game)
end
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
s0_input = MarkovGames.convert_s(Vector{Float32}, s0, game)

model = AZ.load_oracle(joinpath(EXPERIMENT_DIR, "oracle_smoos.jld2"))
load_checkpoint!(model, ANCHOR_ITER)
anchor_outputs = outputs(model, input)
anchor_params = flatten_parameters(model)
anchor_components = (
    shared = flatten_parameters(model.shared),
    critic = flatten_parameters(model.critic),
    regret = flatten_parameters(model.regret_head),
    strategy = flatten_parameters(model.strategy_head),
)

load_checkpoint!(model, last(CHECKPOINTS))
final_outputs = outputs(model, input)

rows = NamedTuple[]
for iter in CHECKPOINTS
    load_checkpoint!(model, iter)
    current = outputs(model, input)
    components = (
        shared = flatten_parameters(model.shared),
        critic = flatten_parameters(model.critic),
        regret = flatten_parameters(model.regret_head),
        strategy = flatten_parameters(model.strategy_head),
    )
    row = (;
        iter,
        s0_value = only(Float64.(AZ.value(model, s0_input))),
        value_mean = mean(current.value),
        value_std = std(current.value),
        value_min = minimum(current.value),
        value_max = maximum(current.value),
        value_rms_from_anchor = rms_distance(current.value, anchor_outputs.value),
        value_rms_from_final = rms_distance(current.value, final_outputs.value),
        regret_positive_l2 = mean_positive_regret_l2(current.regret),
        regret_rms_from_anchor = tuple_rms_distance(current.regret, anchor_outputs.regret),
        regret_rms_from_final = tuple_rms_distance(current.regret, final_outputs.regret),
        strategy_entropy = mean_entropy(current.strategy),
        strategy_kl_from_anchor = mean_kl(current.strategy, anchor_outputs.strategy),
        strategy_kl_from_final = mean_kl(current.strategy, final_outputs.strategy),
        strategy_argmax_anchor = argmax_agreement(current.strategy, anchor_outputs.strategy),
        param_rel_from_anchor = relative_distance(flatten_parameters(model), anchor_params),
        shared_rel_from_anchor = relative_distance(components.shared, anchor_components.shared),
        critic_rel_from_anchor = relative_distance(components.critic, anchor_components.critic),
        regret_head_rel_from_anchor = relative_distance(components.regret, anchor_components.regret),
        strategy_head_rel_from_anchor = relative_distance(components.strategy, anchor_components.strategy),
    )
    push!(rows, row)
end

names = propertynames(first(rows))
matrix = Matrix{Any}(undef, length(rows) + 1, length(names))
matrix[1, :] .= String.(names)
for (i, row) in enumerate(rows), (j, name) in enumerate(names)
    matrix[i + 1, j] = getproperty(row, name)
end

output_path = joinpath(EXPERIMENT_DIR, "checkpoint_drift.csv")
writedlm(output_path, matrix, ',')
foreach(row -> println(row), rows)
println("wrote ", output_path)
