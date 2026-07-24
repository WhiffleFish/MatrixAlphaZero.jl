using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Flux
using JLD2
using Printf
using Random
using Statistics

const DEFAULT_DATASET = joinpath(@__DIR__, "regret_fit_dataset_final_iter.jld2")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "regret_fit_results_final_iter")

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function format_duration(seconds::Real)
    isfinite(seconds) || return "unknown"
    total_seconds = max(0, round(Int, seconds))
    hours, remainder = divrem(total_seconds, 3600)
    minutes, seconds = divrem(remainder, 60)
    return hours > 0 ? "$(hours)h $(minutes)m $(seconds)s" : "$(minutes)m $(seconds)s"
end

softplus_output(x) = Flux.softplus.(x)

function state_network(input_dim, width, output_dim; output_activation=:linear)
    network = Chain(
        Dense(input_dim => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
        Dense(width => output_dim),
    )
    output_activation == :linear && return network
    output_activation == :softplus && return Chain(network.layers..., softplus_output)
    error("Unsupported output activation $(output_activation)")
end

struct HurdleRegressor{T,G,M}
    trunk::T
    gate::G
    log_magnitude::M
end

Flux.@layer :expand HurdleRegressor
Flux.trainable(model::HurdleRegressor) = (;
    trunk=model.trunk,
    gate=model.gate,
    log_magnitude=model.log_magnitude,
)

function HurdleRegressor(input_dim::Integer, width::Integer, output_dim::Integer)
    trunk = Chain(
        Dense(input_dim => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
    )
    return HurdleRegressor(
        trunk,
        Dense(width => output_dim),
        Dense(width => output_dim),
    )
end

stable_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

function hurdle_outputs(model::HurdleRegressor, X, magnitude_scale)
    encoded = model.trunk(X)
    gate_logits = model.gate(encoded)
    predicted_log1p_magnitude = stable_softplus.(model.log_magnitude(encoded))
    magnitude = magnitude_scale .* expm1.(predicted_log1p_magnitude)
    gate_probability = Flux.sigmoid.(gate_logits)
    prediction = gate_probability .* magnitude
    return (;
        gate_logits,
        gate_probability,
        predicted_log1p_magnitude,
        magnitude,
        prediction,
    )
end

function huber_elements(error; delta=1.0f0)
    absolute = abs.(error)
    return ifelse.(
        absolute .<= delta,
        0.5f0 .* error .^ 2,
        delta .* (absolute .- 0.5f0 * delta),
    )
end

huber_mean(error; delta=1.0f0) = mean(huber_elements(error; delta))

function bce_with_logits(logits, labels)
    return mean(
        max.(logits, zero(eltype(logits))) .-
        logits .* labels .+
        log1p.(exp.(-abs.(logits))),
    )
end

effective_target(Y, tau) = ifelse.(Y .> tau, Y, zero(eltype(Y)))

function hurdle_loss(
        model,
        X,
        Y,
        tau;
        gate_weight=1.0f0,
        magnitude_weight=1.0f0,
        reconstruction_weight=0.1f0,
    )
    output = hurdle_outputs(model, X, tau)
    labels = Float32.(Y .> tau)
    gate_loss = bce_with_logits(output.gate_logits, labels)
    target_log1p_magnitude = log1p.(Y ./ tau)
    positive_count = sum(labels)
    magnitude_loss = sum(
        labels .* huber_elements(
            output.predicted_log1p_magnitude .- target_log1p_magnitude,
        ),
    ) / max(positive_count, one(positive_count))
    target = effective_target(Y, tau)
    reconstruction_loss = huber_mean(output.prediction .- target)
    total = gate_weight * gate_loss +
        magnitude_weight * magnitude_loss +
        reconstruction_weight * reconstruction_loss
    return (; total, gate_loss, magnitude_loss, reconstruction_loss)
end

baseline_loss(model, X, Y) = huber_mean(model(X) .- Y)

function minibatches(indices, batch_size)
    return [indices[i:min(i + batch_size - 1, end)] for i in 1:batch_size:length(indices)]
end

function train_model!(
        model,
        loss_function,
        validation_score,
        X,
        Y,
        train_indices,
        validation_indices;
        epochs,
        batch_size,
        learning_rate,
        patience,
        seed,
        label,
        log_every=1,
    )
    rng = MersenneTwister(seed)
    optimiser_state = Flux.setup(Flux.Optimisers.Adam(learning_rate), model)
    best_model = deepcopy(model)
    best_validation = Inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = NamedTuple[]
    started_at = time()

    println(
        "[regret-fit] model=$(label) initialization=fresh_random ",
        "train_samples=$(length(train_indices)) validation_samples=$(length(validation_indices)) ",
        "epochs=$(epochs) batch_size=$(batch_size) lr=$(learning_rate)",
    )
    flush(stdout)

    for epoch in 1:epochs
        epoch_started_at = time()
        shuffled = shuffle(rng, train_indices)
        losses = Float64[]
        for indices in minibatches(shuffled, batch_size)
            loss, gradient = Flux.withgradient(model) do active_model
                loss_function(active_model, X[:, indices], Y[:, indices])
            end
            Flux.update!(optimiser_state, model, gradient[1])
            push!(losses, Float64(loss))
        end
        validation = Float64(validation_score(
            model,
            X[:, validation_indices],
            Y[:, validation_indices],
        ))
        push!(history, (;
            epoch,
            training_loss=mean(losses),
            validation_score=validation,
        ))

        if validation < best_validation
            best_validation = validation
            best_epoch = epoch
            best_model = deepcopy(model)
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end
        if epoch == 1 || iszero(epoch % log_every) || epoch == epochs
            elapsed = time() - started_at
            eta = epoch > 0 ? (epochs - epoch) * elapsed / epoch : Inf
            @printf(
                "[regret-fit] model=%s epoch=%d/%d train=%.6f validation_rmse=%.6f best_epoch=%d stale=%d/%d epoch_seconds=%.2f elapsed=%s max_eta=%s\n",
                label,
                epoch,
                epochs,
                mean(losses),
                sqrt(validation),
                best_epoch,
                epochs_without_improvement,
                patience,
                time() - epoch_started_at,
                format_duration(elapsed),
                format_duration(eta),
            )
            flush(stdout)
        end
        epochs_without_improvement >= patience && break
    end
    return best_model, history, best_epoch, best_validation
end

function safe_divide(numerator, denominator)
    return iszero(denominator) ? NaN : numerator / denominator
end

function prediction_metrics(prediction, target, tau; gate_probability=nothing)
    thresholded = effective_target(target, tau)
    active = target .> tau
    inactive = .!active
    raw_mse = mean(abs2, prediction .- target)
    effective_mse = mean(abs2, prediction .- thresholded)
    zero_mse = mean(abs2, thresholded)
    row = (;
        samples=size(target, 2),
        entries=length(target),
        raw_rmse=sqrt(raw_mse),
        effective_rmse=sqrt(effective_mse),
        skill_vs_zero=1 - safe_divide(effective_mse, zero_mse),
        active_fraction=mean(active),
        active_rmse=any(active) ? sqrt(mean(abs2, prediction[active] .- target[active])) : NaN,
        inactive_rmse=any(inactive) ? sqrt(mean(abs2, prediction[inactive])) : NaN,
        negative_prediction_fraction=mean(prediction .< 0),
    )
    isnothing(gate_probability) && return merge(row, (;
        gate_brier=NaN,
        gate_precision=NaN,
        gate_recall=NaN,
        gate_positive_rate=NaN,
    ))
    labels = active
    predicted_active = gate_probability .>= 0.5
    true_positive = sum(predicted_active .& labels)
    return merge(row, (;
        gate_brier=mean(abs2, gate_probability .- Float32.(labels)),
        gate_precision=safe_divide(true_positive, sum(predicted_active)),
        gate_recall=safe_divide(true_positive, sum(labels)),
        gate_positive_rate=mean(predicted_active),
    ))
end

function metric_row(model_name, player, split, prediction, target, tau; gate_probability=nothing)
    return merge(
        (; model=model_name, player, split),
        prediction_metrics(prediction, target, tau; gate_probability),
    )
end

function csv_value(value)
    value isa AbstractString && return value
    value isa Integer && return string(value)
    value isa Real && return isfinite(value) ? string(value) : ""
    return string(value)
end

function write_metrics_csv(path, rows)
    columns = collect(propertynames(first(rows)))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
    return path
end

function main(args=ARGS)
    test = "--test" in args
    dataset_path = abspath(option_value(args, "--dataset", DEFAULT_DATASET, String))
    output_dir = abspath(option_value(args, "--output-dir", DEFAULT_OUTPUT_DIR, String))
    epochs = option_value(args, "--epochs", 300, x -> parse(Int, x))
    batch_size = option_value(args, "--batch-size", 256, x -> parse(Int, x))
    learning_rate = Float32(option_value(args, "--lr", 3e-4, x -> parse(Float64, x)))
    patience = option_value(args, "--patience", 30, x -> parse(Int, x))
    width = option_value(args, "--width", 64, x -> parse(Int, x))
    baseline_activation = Symbol(lowercase(option_value(
        args,
        "--baseline-activation",
        "linear",
        String,
    )))
    tau = Float32(option_value(args, "--tau", 1e-3, x -> parse(Float64, x)))
    seed = option_value(args, "--seed", 20260722, x -> parse(Int, x))
    reconstruction_weight = Float32(option_value(
        args,
        "--reconstruction-weight",
        0.1,
        x -> parse(Float64, x),
    ))
    log_every = option_value(args, "--log-every", 1, x -> parse(Int, x))
    if test
        epochs = min(epochs, 3)
        patience = min(patience, 3)
        batch_size = min(batch_size, 16)
    end

    isfile(dataset_path) || error(
        "Missing dataset $(dataset_path). Run generate_regret_fit_dataset.jl first.",
    )
    tau > 0 || error("--tau must be positive")
    epochs > 0 || error("--epochs must be positive")
    batch_size > 0 || error("--batch-size must be positive")
    log_every > 0 || error("--log-every must be positive")
    baseline_activation in (:linear, :softplus) || error(
        "--baseline-activation must be linear or softplus",
    )

    data = JLD2.load(dataset_path)
    X = Float32.(data["states"])
    targets = (Float32.(data["regret_p1"]), Float32.(data["regret_p2"]))
    checkpoint_predictions = (
        Float32.(data["checkpoint_regret_p1"]),
        Float32.(data["checkpoint_regret_p2"]),
    )
    train_indices = Int.(data["train_indices"])
    validation_indices = Int.(data["validation_indices"])
    test_indices = Int.(data["test_indices"])
    metadata = data["metadata"]
    isempty(intersect(train_indices, validation_indices)) || error("Train/validation leakage")
    isempty(intersect(train_indices, test_indices)) || error("Train/test leakage")
    isempty(intersect(validation_indices, test_indices)) || error("Validation/test leakage")

    println(
        "[regret-fit] dataset=$(dataset_path) samples=$(size(X, 2)) ",
        "splits=$(length(train_indices))/$(length(validation_indices))/$(length(test_indices))",
    )
    checkpoint_iteration = get(metadata, "checkpoint_iteration", "unknown")
    sample_scope = get(metadata, "sample_scope", "unspecified")
    search_epsilon = get(metadata, "search_epsilon", "unknown")
    action_epsilon = get(metadata, "action_epsilon", "unknown")
    println(
        "[regret-fit] checkpoint_iteration=$(checkpoint_iteration) ",
        "sample_scope=$(sample_scope) search_epsilon=$(search_epsilon) ",
        "action_epsilon=$(action_epsilon)",
    )
    println("[regret-fit] all fitted models use fresh random initialization")
    flush(stdout)

    input_dim = size(X, 1)
    baseline_models = Any[]
    hurdle_models = Any[]
    baseline_histories = Any[]
    hurdle_histories = Any[]
    best_epochs = Dict{String,Int}()

    for player in 1:2
        output_dim = size(targets[player], 1)

        Random.seed!(seed + 100 * player)
        baseline = state_network(
            input_dim,
            width,
            output_dim;
            output_activation=baseline_activation,
        )
        baseline_train_loss = (model, x, y) -> baseline_loss(model, x, y)
        baseline_validation = (model, x, y) -> mean(abs2, model(x) .- y)
        baseline, baseline_history, baseline_epoch, _ = train_model!(
            baseline,
            baseline_train_loss,
            baseline_validation,
            X,
            targets[player],
            train_indices,
            validation_indices;
            epochs,
            batch_size,
            learning_rate,
            patience,
            seed=seed + 1_000 * player,
            label="$(baseline_activation)_baseline_p$(player)",
            log_every,
        )
        push!(baseline_models, baseline)
        push!(baseline_histories, baseline_history)
        best_epochs["baseline_p$(player)"] = baseline_epoch

        Random.seed!(seed + 100 * player)
        hurdle = HurdleRegressor(input_dim, width, output_dim)
        hurdle_train_loss = (model, x, y) -> hurdle_loss(
            model,
            x,
            y,
            tau;
            reconstruction_weight,
        ).total
        hurdle_validation = (model, x, y) -> begin
            prediction = hurdle_outputs(model, x, tau).prediction
            mean(abs2, prediction .- effective_target(y, tau))
        end
        hurdle, hurdle_history, hurdle_epoch, _ = train_model!(
            hurdle,
            hurdle_train_loss,
            hurdle_validation,
            X,
            targets[player],
            train_indices,
            validation_indices;
            epochs,
            batch_size,
            learning_rate,
            patience,
            seed=seed + 1_000 * player,
            label="hurdle_p$(player)",
            log_every,
        )
        push!(hurdle_models, hurdle)
        push!(hurdle_histories, hurdle_history)
        best_epochs["hurdle_p$(player)"] = hurdle_epoch
    end

    rows = NamedTuple[]
    splits = (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    )
    for player in 1:2, (split_name, indices) in splits
        target = targets[player][:, indices]
        checkpoint_prediction = checkpoint_predictions[player][:, indices]
        baseline_prediction = baseline_models[player](X[:, indices])
        hurdle_output = hurdle_outputs(hurdle_models[player], X[:, indices], tau)
        push!(rows, metric_row(
            "checkpoint_single_head",
            player,
            split_name,
            checkpoint_prediction,
            target,
            tau,
        ))
        push!(rows, metric_row(
            "refit_single_head",
            player,
            split_name,
            baseline_prediction,
            target,
            tau,
        ))
        push!(rows, metric_row(
            "hurdle_log_magnitude",
            player,
            split_name,
            hurdle_output.prediction,
            target,
            tau;
            gate_probability=hurdle_output.gate_probability,
        ))
    end

    mkpath(output_dir)
    metrics_path = write_metrics_csv(joinpath(output_dir, "metrics.csv"), rows)
    models_path = joinpath(output_dir, "models.jld2")
    fit_metadata = Dict{String,Any}(
        "dataset" => dataset_path,
        "dataset_metadata" => metadata,
        "tau" => tau,
        "magnitude_transform" => "tau * expm1(softplus(raw))",
        "gate_loss" => "binary_cross_entropy_with_logits",
        "magnitude_loss" => "positive_only_huber_on_log1p_magnitude",
        "reconstruction_loss" => "huber_on_gate_times_magnitude",
        "reconstruction_weight" => reconstruction_weight,
        "epochs" => epochs,
        "batch_size" => batch_size,
        "learning_rate" => learning_rate,
        "patience" => patience,
        "width" => width,
        "baseline_activation" => String(baseline_activation),
        "seed" => seed,
        "initialization" => "fresh_random",
        "log_every" => log_every,
        "best_epochs" => best_epochs,
        "test_mode" => test,
    )
    jldsave(
        models_path;
        baseline_p1_state=Flux.state(baseline_models[1]),
        baseline_p2_state=Flux.state(baseline_models[2]),
        hurdle_p1_state=Flux.state(hurdle_models[1]),
        hurdle_p2_state=Flux.state(hurdle_models[2]),
        baseline_histories,
        hurdle_histories,
        metadata=fit_metadata,
    )

    println("\nHeld-out test results (lower RMSE is better):")
    for row in rows
        row.split == "test" || continue
        @printf(
            "%-24s p%d raw_rmse=%.6f effective_rmse=%.6f skill_vs_zero=%.3f",
            row.model,
            row.player,
            row.raw_rmse,
            row.effective_rmse,
            row.skill_vs_zero,
        )
        if row.model == "hurdle_log_magnitude"
            @printf(
                " gate_precision=%.3f gate_recall=%.3f",
                row.gate_precision,
                row.gate_recall,
            )
        end
        println()
    end
    println("Wrote metrics: $(metrics_path)")
    println("Wrote fitted model states: $(models_path)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
