# Score every available fitted regret model by the quantity the *search*
# actually consumes, rather than by regression RMSE.
#
#   julia --project=experiments experiments/sda/sda-2026-07-21/score_regret_directions.jl
#
# Regret matching normalizes its input: the strategy a warm-started node plays is
# RM([R̄̂]₊) = [R̄̂]₊/‖[R̄̂]₊‖₁, which is invariant to the scale of R̄̂. Every
# deployment knob that multiplies the transferred vector — `prior_scale`, the
# reach attenuation — therefore has no effect at a node whose own accumulated
# regret is still zero, and those are the majority of expanded nodes at a
# 100-query budget. What survives is the *direction*.
#
# So the decision-relevant fit metrics are properties of RM([R̄̂]₊):
#
#   tv         mean total-variation distance to RM([R̄]₊)
#   tv_uniform the same distance for a uniform prior — the do-nothing baseline
#   closed     1 - tv/tv_uniform: fraction of the uniform prior's distance the
#              fitted direction actually closes. This is the number that decides
#              whether transfer can help at all.
#   agree      argmax agreement with RM([R̄]₊)
#   agree_base accuracy of always predicting the most common true argmax
#   support_f1 F1 of recovering the true positive-regret support, which is the
#              part of an RM+ average-regret vector that carries the most
#              information and the part a squared-error fit destroys.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Flux
using JLD2
using Printf
using Statistics

const DATASET = joinpath(@__DIR__, "regret_fit_dataset_final_iter.jld2")

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

function HurdleRegressor(input_dim::Integer, width::Integer, output_dim::Integer)
    trunk = Chain(
        Dense(input_dim => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
    )
    return HurdleRegressor(trunk, Dense(width => output_dim), Dense(width => output_dim))
end

stable_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

function hurdle_outputs(model::HurdleRegressor, X, magnitude_scale)
    encoded = model.trunk(X)
    gate_probability = Flux.sigmoid.(model.gate(encoded))
    magnitude = magnitude_scale .* expm1.(stable_softplus.(model.log_magnitude(encoded)))
    return (; gate_probability, magnitude, prediction=gate_probability .* magnitude)
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
rmpol(v) = (w = max.(Float64.(v), 0.0); s = sum(w); s > 0 ? w ./ s : fill(1 / length(w), length(w)))
tvdist(p, q) = 0.5 * sum(abs, p .- q)

function f1(pred::AbstractMatrix{Bool}, truth::AbstractMatrix{Bool})
    tp = sum(pred .& truth)
    fp = sum(pred .& .!truth)
    fn = sum(.!pred .& truth)
    prec = tp + fp > 0 ? tp / (tp + fp) : 0.0
    rec = tp + fn > 0 ? tp / (tp + fn) : 0.0
    return prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0.0
end

function score(R::AbstractMatrix, Rhat::AbstractMatrix)
    n = size(R, 2)
    na = size(R, 1)
    Ptrue = [rmpol(view(R, :, i)) for i in 1:n]
    Phat = [rmpol(view(Rhat, :, i)) for i in 1:n]
    u = fill(1 / na, na)
    tv = mean(i -> tvdist(Phat[i], Ptrue[i]), 1:n)
    tvu = mean(i -> tvdist(u, Ptrue[i]), 1:n)
    truth_argmax = [argmax(p) for p in Ptrue]
    base = maximum(a -> count(==(a), truth_argmax), 1:na) / n
    return (;
        rmse = sqrt(mean(abs2, Float64.(Rhat) .- Float64.(R))),
        tv, tv_uniform = tvu,
        closed = 1 - tv / tvu,
        agree = mean(i -> argmax(Phat[i]) == truth_argmax[i], 1:n),
        agree_base = base,
        support_f1 = f1(Float64.(Rhat) .> 0, Float64.(R) .> 0),
    )
end

# ---------------------------------------------------------------------------
function top1(M::AbstractMatrix)
    out = zeros(Float32, size(M))
    for j in axes(M, 2)
        out[argmax(view(M, :, j)), j] = 1
    end
    return out
end

# `keep` masks a prediction to a support and, in the `uniform` variant, discards
# the within-support magnitudes as well. If a column's support is empty the
# column is left at zero, which makes the node fall back to the cold uniform
# strategy rather than to an arbitrary direction.
function masked(M::AbstractMatrix, mask::AbstractMatrix; uniform=false)
    kept = uniform ? Float32.(mask) : M .* mask
    return kept
end

function load_predictions(dir, X)
    path = joinpath(@__DIR__, dir, "models.jld2")
    isfile(path) || return nothing
    data = JLD2.load(path)
    meta = data["metadata"]
    width = meta["width"]
    tau = Float32(meta["tau"])
    activation = get(meta, "baseline_activation", "linear")
    input_dim, output_dim = size(X, 1), 3
    out = Dict{String,Any}()
    for p in 1:2
        baseline = state_network(
            input_dim, width, output_dim;
            output_activation=Symbol(activation),
        )
        Flux.loadmodel!(baseline, data["baseline_p$(p)_state"])
        base = baseline(X)
        out["baseline_p$p"] = base
        out["baseline_top1_p$p"] = top1(base)

        hurdle = HurdleRegressor(input_dim, width, output_dim)
        Flux.loadmodel!(hurdle, data["hurdle_p$(p)_state"])
        h = hurdle_outputs(hurdle, X, tau)
        out["hurdle_p$p"] = h.prediction
        # Gate-only direction: the support classifier's probabilities used
        # directly, discarding magnitude.
        out["hurdle_gate_p$p"] = h.gate_probability
        for (tag, θ) in (("03", 0.3f0), ("05", 0.5f0), ("07", 0.7f0))
            mask = h.gate_probability .> θ
            out["hurdle_hard$(tag)_p$p"] = masked(h.magnitude, mask)
            out["hurdle_sup$(tag)_p$p"] = masked(h.magnitude, mask; uniform=true)
            # Magnitude from the better-fitting baseline, support from the gate.
            out["mixed_hard$(tag)_p$p"] = masked(base, mask)
        end
    end
    return out
end

const VARIANTS = (
    "baseline", "baseline_top1", "hurdle", "hurdle_gate",
    "hurdle_hard03", "hurdle_hard05", "hurdle_hard07",
    "hurdle_sup03", "hurdle_sup05", "hurdle_sup07",
    "mixed_hard03", "mixed_hard05", "mixed_hard07",
)

function collect_models(d, indices)
    X = d["states"][:, indices]
    models = Dict{String,NTuple{2,Any}}()
    models["checkpoint_head"] =
        (d["checkpoint_regret_p1"][:, indices], d["checkpoint_regret_p2"][:, indices])
    # Control: the learned average-strategy head reinterpreted as a transfer
    # direction. If it beats the regret heads, the regret head is not the best
    # available prior.
    models["strategy_head"] =
        (d["checkpoint_strategy_p1"][:, indices], d["checkpoint_strategy_p2"][:, indices])
    for dir in ("regret_fit_results_final_iter", "regret_fit_results_softplus_long")
        preds = load_predictions(dir, X)
        isnothing(preds) && continue
        tag = replace(dir, "regret_fit_results_" => "")
        for name in VARIANTS
            haskey(preds, "$(name)_p1") || continue
            models["$(tag)/$(name)"] = (preds["$(name)_p1"], preds["$(name)_p2"])
        end
    end
    return models
end

function report(d, indices, label)
    R = (d["regret_p1"][:, indices], d["regret_p2"][:, indices])
    models = collect_models(d, indices)
    names = sort(collect(keys(models)))
    scored = Dict(name => [score(R[p], models[name][p]) for p in 1:2] for name in names)
    @printf("\n=== %s split (n = %d) ===\n", label, length(indices))
    @printf("%-40s %8s %8s %8s %8s %8s %8s\n",
            "model", "rmse1", "rmse2", "closed1", "closed2", "argmax1", "argmax2")
    for name in sort(names; by=n -> -(scored[n][1].closed + scored[n][2].closed))
        s1, s2 = scored[name]
        @printf("%-40s %8.4f %8.4f %7.1f%% %7.1f%% %8.3f %8.3f\n",
                name, s1.rmse, s2.rmse, 100 * s1.closed, 100 * s2.closed, s1.agree, s2.agree)
    end
    return scored
end

function main()
    d = JLD2.load(DATASET)
    val = report(d, d["validation_indices"], "validation")
    test = report(d, d["test_indices"], "test")

    best = argmax(n -> val[n][1].closed + val[n][2].closed, collect(keys(val)))
    deployed = "softplus_long/baseline"
    println()
    @printf("selected on validation:  %s  -> test closed = %.1f%% / %.1f%%\n",
            best, 100 * test[best][1].closed, 100 * test[best][2].closed)
    @printf("currently deployed:      %s  -> test closed = %.1f%% / %.1f%%\n",
            deployed, 100 * test[deployed][1].closed, 100 * test[deployed][2].closed)
    println()
    println("`closed` is the fraction of a uniform prior's total-variation distance")
    println("to the true regret-matching direction that the fitted direction closes.")
    println("It, not rmse, determines whether the transferred prior can help: regret")
    println("matching normalizes its input, so only the direction reaches the search.")
    @printf("argmax base rate (always predict the most common true argmax): %.3f / %.3f\n",
            test["checkpoint_head"][1].agree_base, test["checkpoint_head"][2].agree_base)
end

main()
