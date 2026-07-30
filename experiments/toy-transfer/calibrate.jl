# Measure the SDA regret head's error profile on its held-out test split, then
# fit the toy oracle's `NoiseSpec` so the toy reproduces it.
#
#   julia --project=. experiments/toy-transfer/calibrate.jl
#
# The statistics that matter for the search are properties of the *induced*
# regret-matching direction, because regret matching normalizes away the scale
# of the transferred vector:
#
#   tv   = mean total-variation distance between RM(R̄̂) and RM(R̄)
#   ent  = mean entropy of RM(R̄̂)
#   agree= fraction of states where RM(R̄̂) and RM(R̄) share an argmax
#
# The reference point is `tv_uniform`, the same distance for a uniform prior: if
# tv ≈ tv_uniform the transferred direction carries almost no usable signal.
using JLD2
using Statistics
using Printf
using Random

const SDA_DATASET = joinpath(
    @__DIR__, "..", "sda", "sda-2026-07-21", "regret_fit_dataset_final_iter.jld2",
)

rmpol(v) = (w = max.(Float64.(v), 0.0); s = sum(w); s > 0 ? w ./ s : fill(1 / length(w), length(w)))
entropy(p) = -sum(x -> x > 0 ? x * log(x) : 0.0, p)
tv(p, q) = 0.5 * sum(abs, p .- q)

function direction_stats(R::AbstractMatrix, Rhat::AbstractMatrix)
    Ptrue = [rmpol(view(R, :, i)) for i in axes(R, 2)]
    Phat = [rmpol(view(Rhat, :, i)) for i in axes(Rhat, 2)]
    u = fill(1 / size(R, 1), size(R, 1))
    return (;
        signal = mean(maximum(abs, R; dims=1)),
        delta_R = sqrt(mean(abs2, Rhat .- R)),
        snr = mean(maximum(abs, R; dims=1)) / sqrt(mean(abs2, Rhat .- R)),
        tv = mean(i -> tv(Phat[i], Ptrue[i]), eachindex(Phat)),
        tv_uniform = mean(i -> tv(u, Ptrue[i]), eachindex(Ptrue)),
        ent = mean(entropy, Phat),
        ent_true = mean(entropy, Ptrue),
        agree = mean(i -> argmax(Phat[i]) == argmax(Ptrue[i]), eachindex(Phat)),
        npos_true = mean(sum(R .> 0; dims=1)),
        npos_pred = mean(sum(Rhat .> 0; dims=1)),
    )
end

function measure_sda()
    isfile(SDA_DATASET) || error("Missing SDA dataset: $(SDA_DATASET)")
    d = load(SDA_DATASET)
    test = d["test_indices"]
    return map(1:2) do p
        direction_stats(d["regret_p$p"][:, test], d["checkpoint_regret_p$p"][:, test])
    end
end

# Simulate the toy's noise model on synthetic sparse nonnegative regret vectors
# with the same positivity structure as the SDA targets, and report the same
# statistics.
function toy_stats(σ_r, α_r; n=20_000, npos_true=1.7, seed=7)
    rng = MersenneTwister(seed)
    tvs = Float64[]; ents = Float64[]; agrees = Bool[]; tvus = Float64[]
    u = fill(1 / 3, 3)
    for _ in 1:n
        # sparse nonnegative target: 1 or 2 positive components
        k = rand(rng) < (npos_true - 1) ? 2 : 1
        R = zeros(3)
        for a in randperm(rng, 3)[1:k]
            R[a] = abs(randn(rng))
        end
        base = rmpol(R)
        p = (1 - α_r) .* base .+ α_r / 3 .+ σ_r .* abs.(randn(rng, 3))
        p ./= sum(p)
        push!(tvs, tv(p, base))
        push!(ents, entropy(p))
        push!(agrees, argmax(p) == argmax(base))
        push!(tvus, tv(u, base))
    end
    return (; tv=mean(tvs), ent=mean(ents), agree=mean(agrees), tv_uniform=mean(tvus))
end

function fit_noise(target; σ_grid=0.0:0.05:6.0, α_grid=0.0:0.05:1.0)
    best = nothing
    for σ_r in σ_grid, α_r in α_grid
        s = toy_stats(σ_r, α_r)
        loss = (s.tv - target.tv)^2 + (s.ent - target.ent)^2
        if isnothing(best) || loss < best.loss
            best = (; loss, σ_r, α_r, s)
        end
    end
    return best
end

function main()
    stats = measure_sda()
    println("=== measured SDA regret-fit error (held-out test split) ===")
    for (p, s) in enumerate(stats)
        @printf("player %d:\n", p)
        @printf("  mean |R̄|_inf (signal) = %.4f   delta_R (RMSE) = %.4f   SNR = %.2f\n",
                s.signal, s.delta_R, s.snr)
        @printf("  positive components: true %.2f  predicted %.2f  (of 3)\n",
                s.npos_true, s.npos_pred)
        @printf("  entropy RM(R̄)  = %.4f    entropy RM(R̄̂) = %.4f   (max %.4f)\n",
                s.ent_true, s.ent, log(3))
        @printf("  TV(RM(R̄̂),RM(R̄)) = %.4f   vs TV(uniform,RM(R̄)) = %.4f\n",
                s.tv, s.tv_uniform)
        @printf("  argmax agreement = %.3f\n", s.agree)
        @printf("  --> the fitted direction recovers %.0f%% of the distance a uniform prior leaves\n",
                100 * (1 - s.tv / s.tv_uniform))
    end

    println("\n=== fitted toy NoiseSpec (matches TV and entropy of RM(R̄̂)) ===")
    for (p, s) in enumerate(stats)
        best = fit_noise(s)
        @printf("player %d: σ_r = %.2f  α_r = %.2f  -> tv %.4f (target %.4f)  ent %.4f (target %.4f)  agree %.3f (target %.3f)\n",
                p, best.σ_r, best.α_r, best.s.tv, s.tv, best.s.ent, s.ent, best.s.agree, s.agree)
    end
    println("\nUse the pooled setting in run.jl as NoiseSpec(σ_r=..., α_r=...).")
end

main()
