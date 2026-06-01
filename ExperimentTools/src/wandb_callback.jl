struct WandbCallback
    logger::WandbLogger
end

function WandbCallback(; project::String, name::Union{String,Nothing}=nothing, group::Union{String,Nothing}=nothing, config::AbstractDict=Dict{String,Any}())
    return WandbCallback(WandbLogger(; project, name, group, config))
end

function (cb::WandbCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    metrics = Dict{String, Any}()
    for k in propertynames(info)
        k in (:oracle, :online_oracle, :ema_oracle) && continue
        v = getproperty(info, k)
        if v isa Number && isfinite(v)
            metrics[wandb_metric_key(k)] = v
        elseif k == :minibatch_metrics
            metrics[wandb_metric_key(k)] = minibatch_metrics_table(v)
        end
    end
    Wandb.log(cb.logger, metrics; step=info.iter)
end

Base.close(cb::WandbCallback) = close(cb.logger)

const WANDB_TRAINING_HEALTH_KEYS = Set((
    :mean_loss,
    :mean_value_loss,
    :mean_regret_loss,
    :mean_strategy_loss,
    :mean_grad_norm,
    :max_grad_norm,
    :minibatch_metrics,
))

const WANDB_ORACLE_QUALITY_KEYS = Set((
    :value_pred_mse,
    :regret_pred_mse,
    :strategy_entropy_p1,
    :strategy_entropy_p2,
    :strategy_kl_p1,
    :strategy_kl_p2,
    :target_strategy_kl_p1,
    :target_strategy_kl_p2,
))

const WANDB_SELFPLAY_KEYS = Set((
    :mean_ep_length,
    :mean_reward,
    :reward_std,
    :mean_search_time,
    :total_search_time,
    :search_count,
    :batch_size,
))

const WANDB_PROGRESS_KEYS = Set((
    :iter,
    :update,
    :steps_done,
    :max_steps,
    :samples_added,
    :exploration_epsilon,
))

function wandb_metric_key(k::Symbol)
    k in WANDB_TRAINING_HEALTH_KEYS && return "training_health/$(k)"
    k in WANDB_ORACLE_QUALITY_KEYS && return "oracle_quality/$(k)"
    k in WANDB_SELFPLAY_KEYS && return "selfplay/$(k)"
    k in WANDB_PROGRESS_KEYS && return "progress/$(k)"
    return string(k)
end

function minibatch_metrics_table(metrics)
    columns = ["minibatch", "loss", "value_loss", "regret_loss", "strategy_loss", "grad_norm"]
    n = length(metrics.minibatch)
    data = Matrix{Float64}(undef, n, length(columns))
    data[:, 1] .= Float64.(metrics.minibatch)
    data[:, 2] .= Float64.(metrics.loss)
    data[:, 3] .= Float64.(metrics.value_loss)
    data[:, 4] .= Float64.(metrics.regret_loss)
    data[:, 5] .= Float64.(metrics.strategy_loss)
    data[:, 6] .= Float64.(metrics.grad_norm)
    return Wandb.Table(; data, columns)
end

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
    group   :: String
    state   :: String
    config  :: Dict{String, Any}
    metrics :: Dict{String, Vector{Float64}}
end

Base.show(io::IO, r::WandbRun) =
    print(io, "WandbRun(\"$(r.name)\", group=$(r.group), state=$(r.state), metrics=$(sort(collect(keys(r.metrics)))))")

"""
    fetch_wandb_runs(project; entity=nothing, filters=Dict()) -> Vector{WandbRun}

Pull metric history for all runs in a W&B project via the W&B public API and return them
as a `Vector{WandbRun}` for analysis in Julia.

# Arguments
- `project`: W&B project name (e.g. `"Matrix AlphaZero"`).
- `entity`: W&B username or team name. If `nothing`, defaults to the authenticated user.
- `filters`: MongoDB-style filter dict forwarded to the W&B API
  (e.g. `Dict("config.search_style" => "regret_matching")`).

# Example
```julia
runs = fetch_wandb_runs("Matrix AlphaZero")

# Filter to a specific search style
rm_runs = filter(r -> get(r.config, "search_style", "") == "regret_matching", runs)

# Extract a metric time series
losses = mg_runs[1].metrics["training_health/mean_loss"]

# Compare value MSE across styles
for r in runs
    println(r.name, ": final value_mse = ", last(r.metrics["oracle_quality/value_pred_mse"]))
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

        # run.config is a W&B Config object (dict subclass); pyconvert needs a
        # plain Python dict, otherwise it silently produces an empty Julia dict.
        config_dict = PythonCall.pydict(run.config)

        # group is a top-level run attribute (not inside config); may be None.
        group_str = something(PythonCall.pyconvert(Union{Nothing, String}, run.group), "")

        WandbRun(
            PythonCall.pyconvert(String, run.id),
            PythonCall.pyconvert(String, run.name),
            group_str,
            PythonCall.pyconvert(String, run.state),
            PythonCall.pyconvert(Dict{String, Any}, config_dict),
            metrics,
        )
    end
end
