using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Flux
using JLD2
using Printf
using Random
using Statistics

const DEFAULT_DATASET = joinpath(@__DIR__, "regret_fit_dataset_final_iter.jld2")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "regret_fit_results_softplus_long")

function option_value(args, name, default, parse_value=identity)
    index = findfirst(==(name), args)
    isnothing(index) && return default
    index < length(args) || error("Missing value after $(name)")
    return parse_value(args[index + 1])
end

softplus_output(x) = Flux.softplus.(x)

function regret_network(input_dim, width, output_dim)
    return Chain(
        Dense(input_dim => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
        Dense(width => output_dim),
        softplus_output,
    )
end

function huber_mean(error; delta=1.0f0)
    absolute = abs.(error)
    return mean(ifelse.(
        absolute .<= delta,
        0.5f0 .* error .^ 2,
        delta .* (absolute .- 0.5f0 * delta),
    ))
end

function train_regressor(
        model,
        X,
        Y,
        train_indices,
        validation_indices;
        epochs,
        batch_size,
        learning_rate,
        patience,
        seed,
        player,
    )
    rng = MersenneTwister(seed)
    optimiser = Flux.setup(Flux.Optimisers.Adam(learning_rate), model)
    best_model = deepcopy(model)
    best_validation = Inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in 1:epochs
        shuffled = shuffle(rng, train_indices)
        for start in 1:batch_size:length(shuffled)
            indices = shuffled[start:min(start + batch_size - 1, end)]
            _, gradient = Flux.withgradient(model) do active_model
                huber_mean(active_model(X[:, indices]) .- Y[:, indices])
            end
            Flux.update!(optimiser, model, gradient[1])
        end

        prediction = model(X[:, validation_indices])
        validation = Float64(mean(abs2, prediction .- Y[:, validation_indices]))
        if validation < best_validation
            best_model = deepcopy(model)
            best_validation = validation
            best_epoch = epoch
            stale_epochs = 0
        else
            stale_epochs += 1
        end
        if epoch == 1 || iszero(epoch % 25) || epoch == epochs
            @printf(
                "[regret-fit] p%d epoch=%d/%d validation_rmse=%.6f best_epoch=%d stale=%d/%d\n",
                player,
                epoch,
                epochs,
                sqrt(validation),
                best_epoch,
                stale_epochs,
                patience,
            )
        end
        stale_epochs >= patience && break
    end
    return best_model, best_epoch
end

function metric_rows(models, checkpoint_predictions, targets, X, splits)
    rows = NamedTuple[]
    for player in 1:2, (split, indices) in splits
        target = targets[player][:, indices]
        zero_mse = mean(abs2, target)
        for (model_name, prediction) in (
                ("checkpoint", checkpoint_predictions[player][:, indices]),
                ("softplus_refit", models[player](X[:, indices])),
            )
            mse = mean(abs2, prediction .- target)
            push!(rows, (;
                model=model_name,
                player,
                split,
                samples=length(indices),
                rmse=sqrt(mse),
                skill_vs_zero=iszero(zero_mse) ? NaN : 1 - mse / zero_mse,
                negative_prediction_fraction=mean(prediction .< 0),
            ))
        end
    end
    return rows
end

csv_value(value::Real) = isfinite(value) ? string(value) : ""
csv_value(value) = string(value)

function write_csv(path, rows)
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join(
                (csv_value(getproperty(row, column)) for column in columns),
                ',',
            ))
        end
    end
end

function main(args=ARGS)
    test = "--test" in args
    dataset_path = abspath(option_value(args, "--dataset", DEFAULT_DATASET, String))
    output_dir = abspath(option_value(args, "--output-dir", DEFAULT_OUTPUT_DIR, String))
    epochs = option_value(args, "--epochs", 1000, value -> parse(Int, value))
    batch_size = option_value(args, "--batch-size", 256, value -> parse(Int, value))
    learning_rate = Float32(option_value(args, "--lr", 3e-4, value -> parse(Float64, value)))
    patience = option_value(args, "--patience", 150, value -> parse(Int, value))
    width = option_value(args, "--width", 64, value -> parse(Int, value))
    seed = option_value(args, "--seed", 20260722, value -> parse(Int, value))
    if test
        epochs = min(epochs, 3)
        patience = min(patience, 3)
        batch_size = min(batch_size, 16)
        output_dir = joinpath(output_dir, "smoke")
    end

    isfile(dataset_path) || error(
        "Missing dataset $(dataset_path). Run generate_regret_fit_dataset.jl first.",
    )
    all(>(0), (epochs, batch_size, patience, width)) ||
        error("epochs, batch size, patience, and width must be positive")

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
    isempty(intersect(train_indices, validation_indices)) || error("Train/validation leakage")
    isempty(intersect(train_indices, test_indices)) || error("Train/test leakage")
    isempty(intersect(validation_indices, test_indices)) || error("Validation/test leakage")

    models = Any[]
    best_epochs = Int[]
    for player in 1:2
        Random.seed!(seed + 100 * player)
        model = regret_network(size(X, 1), width, size(targets[player], 1))
        model, best_epoch = train_regressor(
            model,
            X,
            targets[player],
            train_indices,
            validation_indices;
            epochs,
            batch_size,
            learning_rate,
            patience,
            seed=seed + 1000 * player,
            player,
        )
        push!(models, model)
        push!(best_epochs, best_epoch)
    end

    rows = metric_rows(
        models,
        checkpoint_predictions,
        targets,
        X,
        (("train", train_indices), ("validation", validation_indices), ("test", test_indices)),
    )
    mkpath(output_dir)
    write_csv(joinpath(output_dir, "metrics.csv"), rows)
    metadata = Dict{String,Any}(
        "dataset" => dataset_path,
        "dataset_metadata" => data["metadata"],
        "epochs" => epochs,
        "batch_size" => batch_size,
        "learning_rate" => learning_rate,
        "patience" => patience,
        "width" => width,
        "activation" => "softplus",
        "seed" => seed,
        "best_epochs" => best_epochs,
        "test_mode" => test,
    )
    jldsave(
        joinpath(output_dir, "models.jld2");
        baseline_p1_state=Flux.state(models[1]),
        baseline_p2_state=Flux.state(models[2]),
        metadata,
    )
    println("Wrote fitted models and metrics to $(output_dir)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
