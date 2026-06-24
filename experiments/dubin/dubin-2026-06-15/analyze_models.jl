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
const CHECKPOINTS = 0:1221
const SELECTED_CHECKPOINTS = Set((0, 100, 250, 500, 750, 1000, 1221))
const STATE_COUNT = 2048

function fixed_state_suite(game, n::Int; seed::Int=20260615)
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

function flatten_parameters(model)
    params = Flux.trainables(model)
    isempty(params) && return Float64[]
    return reduce(vcat, (vec(Float64.(p)) for p in params))
end

relative_distance(current, reference) =
    norm(current .- reference) / max(norm(reference), eps(Float64))

function mean_entropy(policy)
    total = 0.0
    for p in policy
        p > 0 && (total -= p * log(p))
    end
    return total / size(policy, 2)
end

function mean_kl(policy, reference)
    total = 0.0
    n = size(policy, 2)
    for j in axes(policy, 2), i in axes(policy, 1)
        p = policy[i, j]
        p > 0 && (total += p * log(p / max(reference[i, j], eps(eltype(reference)))))
    end
    return total / n
end

function argmax_agreement(policy, reference)
    matches = count(j -> argmax(view(policy, :, j)) == argmax(view(reference, :, j)), axes(policy, 2))
    return matches / size(policy, 2)
end

function checkpoint_path(models_dir::String, iter::Int)
    return joinpath(models_dir, "oracle$(AZ.iter2string(iter)).jld2")
end

function model_components(model::AZ.FittedRegretModel)
    return (
        shared = model.shared,
        critic = model.critic,
        head1 = model.regret_head,
        head2 = model.strategy_head,
    )
end

function model_components(model::AZ.ActorCritic)
    return (
        shared = model.shared,
        critic = model.critic,
        head1 = model.actor,
        head2 = nothing,
    )
end

function component_vectors(model)
    components = model_components(model)
    return (
        shared = flatten_parameters(components.shared),
        critic = flatten_parameters(components.critic),
        head1 = flatten_parameters(components.head1),
        head2 = isnothing(components.head2) ? Float64[] : flatten_parameters(components.head2),
    )
end

function policy_metrics(model::AZ.ActorCritic, input, final_policy)
    policy = AZ.policy(model, input)
    return (
        entropy_p1 = mean_entropy(policy[1]),
        entropy_p2 = mean_entropy(policy[2]),
        kl_final_p1 = mean_kl(policy[1], final_policy[1]),
        kl_final_p2 = mean_kl(policy[2], final_policy[2]),
        argmax_final_p1 = argmax_agreement(policy[1], final_policy[1]),
        argmax_final_p2 = argmax_agreement(policy[2], final_policy[2]),
    )
end

function policy_metrics(::AZ.FittedRegretModel, input, final_policy)
    return (
        entropy_p1 = NaN,
        entropy_p2 = NaN,
        kl_final_p1 = NaN,
        kl_final_p2 = NaN,
        argmax_final_p1 = NaN,
        argmax_final_p2 = NaN,
    )
end

function scan_model(method::String, oracle_file::String, models_dir::String, input, s0_input)
    model = AZ.load_oracle(oracle_file)

    Flux.loadmodel!(model, checkpoint_path(models_dir, first(CHECKPOINTS)))
    initial_params = flatten_parameters(model)
    initial_components = component_vectors(model)

    Flux.loadmodel!(model, checkpoint_path(models_dir, last(CHECKPOINTS)))
    final_values = vec(Float64.(AZ.value(model, input)))
    final_policy = model isa AZ.ActorCritic ? AZ.policy(model, input) : nothing

    rows = NamedTuple[]
    previous_values = nothing
    previous_params = nothing

    for iter in CHECKPOINTS
        Flux.loadmodel!(model, checkpoint_path(models_dir, iter))
        values = vec(Float64.(AZ.value(model, input)))
        params = flatten_parameters(model)
        components = component_vectors(model)
        policy = policy_metrics(model, input, final_policy)

        row = (;
            method,
            iter,
            s0_value = only(Float64.(AZ.value(model, s0_input))),
            value_mean = mean(values),
            value_std = std(values),
            value_min = minimum(values),
            value_max = maximum(values),
            value_abs_gt_1 = mean(abs.(values) .> 1),
            value_rms_step = isnothing(previous_values) ? 0.0 : sqrt(mean(abs2, values .- previous_values)),
            value_rms_final = sqrt(mean(abs2, values .- final_values)),
            value_corr_final = std(values) > eps() ? cor(values, final_values) : NaN,
            param_rel_initial = relative_distance(params, initial_params),
            param_rel_step = isnothing(previous_params) ? 0.0 : relative_distance(params, previous_params),
            shared_rel_initial = relative_distance(components.shared, initial_components.shared),
            critic_rel_initial = relative_distance(components.critic, initial_components.critic),
            head1_rel_initial = relative_distance(components.head1, initial_components.head1),
            head2_rel_initial = isempty(components.head2) ? NaN : relative_distance(components.head2, initial_components.head2),
            policy...,
        )
        push!(rows, row)
        previous_values = values
        previous_params = params

        if iter in SELECTED_CHECKPOINTS
            println(
                method,
                " iter=", iter,
                " s0=", round(row.s0_value; digits=4),
                " value_mu=", round(row.value_mean; digits=4),
                " value_sd=", round(row.value_std; digits=4),
                " rms_final=", round(row.value_rms_final; digits=4),
                " corr_final=", round(row.value_corr_final; digits=4),
                " param_delta=", round(row.param_rel_initial; digits=4),
                model isa AZ.ActorCritic ? " policy_kl_final=$(round((row.kl_final_p1 + row.kl_final_p2) / 2; digits=4))" : "",
            )
        end
    end
    return rows, final_values
end

function write_rows(path::String, rows)
    names = propertynames(first(rows))
    matrix = Matrix{Any}(undef, length(rows) + 1, length(names))
    matrix[1, :] .= String.(names)
    for (i, row) in enumerate(rows), (j, name) in enumerate(names)
        matrix[i + 1, j] = getproperty(row, name)
    end
    writedlm(path, matrix, ',')
end

game = DubinMG(V = (1.0, 1.0))
states = fixed_state_suite(game, STATE_COUNT)
input = mapreduce(hcat, states) do state
    MarkovGames.convert_s(Vector{Float32}, state, game)
end
s0 = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
s0_input = MarkovGames.convert_s(Vector{Float32}, s0, game)

smoos_rows, smoos_final_values = scan_model(
    "smoos",
    joinpath(EXPERIMENT_DIR, "oracle_smoos.jld2"),
    joinpath(EXPERIMENT_DIR, "models_smoos"),
    input,
    s0_input,
)
rm_rows, rm_final_values = scan_model(
    "regret_matching",
    joinpath(EXPERIMENT_DIR, "oracle_regret_matching.jld2"),
    joinpath(EXPERIMENT_DIR, "models_regret_matching"),
    input,
    s0_input,
)

output_path = isempty(ARGS) ? joinpath(EXPERIMENT_DIR, "model_analysis.csv") : first(ARGS)
write_rows(output_path, vcat(smoos_rows, rm_rows))
println("final critic correlation smoos_vs_regret_matching=", round(cor(smoos_final_values, rm_final_values); digits=4))
println("wrote ", output_path)
