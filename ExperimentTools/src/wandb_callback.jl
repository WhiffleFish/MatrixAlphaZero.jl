struct WandbCallback
    logger::WandbLogger
end

function WandbCallback(; project::String, name::Union{String,Nothing}=nothing, config::AbstractDict=Dict{String,Any}())
    return WandbCallback(WandbLogger(; project, name, config))
end

function (cb::WandbCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    metrics = Dict{String, Any}()
    for k in propertynames(info)
        k in (:oracle, :online_oracle, :ema_oracle) && continue
        v = getproperty(info, k)
        v isa Number && isfinite(v) && (metrics[string(k)] = v)
    end
    Wandb.log(cb.logger, metrics; step=info.iter)
end

Base.close(cb::WandbCallback) = close(cb.logger)

# ── Fetching runs back for analysis ───────────────────────────────────────────

"""
    WandbRun

Holds the data for a single W&B run returned by [`fetch_wandb_runs`](@ref).

| Field     | Type                        | Description                                      |
|:----------|:----------------------------|:-------------------------------------------------|
| `id`      | `String`                    | Unique run identifier assigned by W&B.           |
| `name`    | `String`                    | Human-readable run name.                         |
| `state`   | `String`                    | Run state: `"finished"`, `"running"`, `"crashed"`|
| `config`  | `Dict{String,Any}`          | Hyperparameters logged at run creation.          |
| `metrics` | `Dict{String,Vector}`       | Per-step metric history, keyed by metric name.   |
|           |                             | Each vector is ordered by training step.         |
"""
struct WandbRun
    id      :: String
    name    :: String
    state   :: String
    config  :: Dict{String, Any}
    metrics :: Dict{String, Vector{Float64}}
end

Base.show(io::IO, r::WandbRun) =
    print(io, "WandbRun(\"$(r.name)\", state=$(r.state), metrics=$(sort(collect(keys(r.metrics)))))")

"""
    fetch_wandb_runs(project; entity=nothing, filters=Dict()) -> Vector{WandbRun}

Pull metric history for all runs in a W&B project via the W&B public API and return them
as a `Vector{WandbRun}` for analysis in Julia.

# Arguments
- `project`: W&B project name (e.g. `"Matrix AlphaZero"`).
- `entity`: W&B username or team name. If `nothing`, defaults to the authenticated user.
- `filters`: MongoDB-style filter dict forwarded to the W&B API
  (e.g. `Dict("config.search_style" => "matrix_game")`).

# Example
```julia
runs = fetch_wandb_runs("Matrix AlphaZero")

# Filter to a specific search style
mg_runs = filter(r -> get(r.config, "search_style", "") == "matrix_game", runs)

# Extract a metric time series
losses = mg_runs[1].metrics["mean_loss"]

# Compare value MSE across styles
for r in runs
    println(r.name, ": final value_mse = ", last(r.metrics["value_pred_mse"]))
end
```
"""
function fetch_wandb_runs(
        project :: String;
        entity  :: Union{String, Nothing} = nothing,
        filters :: Dict = Dict(),
    )
    wandb_py = PythonCall.pyimport("wandb")
    api      = wandb_py.Api()
    path     = isnothing(entity) ? project : "$entity/$project"
    py_runs  = api.runs(path; filters = PythonCall.pydict(filters))

    return map(collect(py_runs)) do run
        # pandas=false returns a list of row-dicts, which works without pandas installed
        rows = PythonCall.pyconvert(
            Vector{Dict{String, Any}},
            run.history(samples=10_000, pandas=false),
        )

        # Collect all non-internal keys across all rows
        cols = String[]
        for row in rows, k in keys(row)
            startswith(k, "_") || k in cols || push!(cols, k)
        end

        # Build per-column Float64 vectors; Python None and non-numeric values → NaN
        to_f64(v) = v isa Number ? Float64(v) : NaN64
        metrics = Dict{String, Vector{Float64}}(
            col => [to_f64(get(row, col, nothing)) for row in rows]
            for col in cols
        )

        WandbRun(
            PythonCall.pyconvert(String, run.id),
            PythonCall.pyconvert(String, run.name),
            PythonCall.pyconvert(String, run.state),
            PythonCall.pyconvert(Dict{String, Any}, run.config),
            metrics,
        )
    end
end
