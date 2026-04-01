# Requires `Wandb` to be added to the ExperimentTools project:
#   using Pkg; Pkg.activate("ExperimentTools"); Pkg.add("Wandb")
# and the Python wandb package:
#   using CondaPkg; CondaPkg.add("wandb")  # or: pip install wandb

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
