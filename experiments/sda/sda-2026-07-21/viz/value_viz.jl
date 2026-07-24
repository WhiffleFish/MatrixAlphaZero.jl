using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using MarkovGames
using MatrixAlphaZero
const AZ = MatrixAlphaZero
using Flux
using JLD2
using POMDPTools
using Distributions
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics
using LinearAlgebra
using Random
using Statistics
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

const expdir = joinpath(@__DIR__, "..")
include(joinpath(expdir, "initial_state.jl"))
const figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
oracle = AZ.load_oracle(joinpath(expdir, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
Flux.loadmodel!(
    oracle,
    JLD2.load(joinpath(expdir, "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"],
)

# ---------------------------------------------------------------------------
# Builders: construct an SDAState2D from physical, interpretable coordinates.
#   target_alt_km   : target altitude (km)
#   rel_alt_km      : observer altitude - target altitude (km)
#   sep_deg         : observer phase - target phase (deg, signed)
#   target_phase    : target mean anomaly / orbital position (rad)
# Both satellites on circular coplanar orbits (velocity fixed by altitude).
# ---------------------------------------------------------------------------
function make_state(; target_alt_km, rel_alt_km, sep_deg, target_phase)
    r_t = R_EARTH + target_alt_km * 1e3
    r_o = R_EARTH + (target_alt_km + rel_alt_km) * 1e3
    φ_o = mod2pi(target_phase + deg2rad(sep_deg))
    target   = SNRGame.sOSCtoCART2D([r_t, 0.0, 0.0, target_phase])
    observer = SNRGame.sOSCtoCART2D([r_o, 0.0, 0.0, φ_o])
    return SNRGame.SDAState2D(observer, target, game.epc0, false)
end

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
values_of(states) = vec(Float64.(AZ.value(oracle, reduce(hcat, feat.(states)))))
value_of(s) = only(values_of([s]))

# no-burn (coast) Monte-Carlo discounted return, player 1 (observer). This is the
# myopic/greedy-detector baseline: what an agent gets if nobody maneuvers.
function noburn_return(s; depth=50)
    γ = MarkovGames.discount(game); G = 0.0; disc = 1.0
    for _ in 1:depth
        MarkovGames.isterminal(game, s) && break
        sp = rand(MarkovGames.transition(game, s, (0.0, 0.0)))
        r = MarkovGames.reward(game, s, (0.0, 0.0), sp)[1]
        G += disc * r; disc *= γ; s = sp
    end
    return G
end

# ===========================================================================
# IDEA 1 — Feature-isolation (partial dependence) panels.
# Sweep ONE physical quantity, hold the rest fixed, plot learned value.
# ===========================================================================
begin
    # (a) value vs orbital position relative to the (fixed) Sun
    φs = LinRange(0, 2π, 200)
    va = [value_of(make_state(target_alt_km=900, rel_alt_km=0, sep_deg=40, target_phase=φ)) for φ in φs]
    pa = plot(rad2deg.(φs), va, lw=3, xlabel="target orbital position  (deg from +x = Sun)",
              ylabel="learned value  V", title="(a) illumination / Sun geometry")
    vline!(pa, [180], ls=:dash, c=:gray)   # anti-Sun (Earth-shadow side)

    # (b) value vs relative altitude (orbit-matching)
    ras = LinRange(-1500, 1500, 200)
    # average over orbital position to marginalize illumination
    φsamp = LinRange(0, 2π, 24)
    vb = [mean(value_of(make_state(target_alt_km=900, rel_alt_km=ra, sep_deg=20, target_phase=φ)) for φ in φsamp) for ra in ras]
    pb = plot(ras, vb, lw=3, xlabel="relative altitude  Δa  (km)",
              ylabel="mean value  V", title="(b) co-orbital matching")
    vline!(pb, [0], ls=:dash, c=:gray)

    # (c) value vs phase separation
    seps = LinRange(-180, 180, 200)
    vc = [mean(value_of(make_state(target_alt_km=900, rel_alt_km=0, sep_deg=sp, target_phase=φ)) for φ in φsamp) for sp in seps]
    pc = plot(seps, vc, lw=3, xlabel="phase separation  Δν  (deg)",
              ylabel="mean value  V", title="(c) angular separation")
    vline!(pc, [0], ls=:dash, c=:gray)

    # (d) value vs absolute altitude regime (independent of relative geometry)
    alts = LinRange(400, 1600, 200)
    vd = [mean(value_of(make_state(target_alt_km=a, rel_alt_km=0, sep_deg=20, target_phase=φ)) for φ in φsamp) for a in alts]
    pd = plot(alts, vd, lw=3, xlabel="target altitude  (km)",
              ylabel="mean value  V", title="(d) altitude regime")
    vspan!(pd, [600, 1200], c=:gray, alpha=0.12, label="train support")

    plot(pa, pb, pc, pd, layout=(2,2), size=(1100, 800), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    savefig(joinpath(figdir, "idea1_partial_dependence.png"))
    println("wrote idea1_partial_dependence.png")
end

# ===========================================================================
# IDEA 2 — Relative orbit-element value map (Δalt x phase-sep), the natural
# competitive coordinate. Marginalized over orbital position.
# ===========================================================================
begin
    sep_ax = LinRange(-140, 140, 90)
    alt_ax = LinRange(-500, 500, 90)
    φsamp = LinRange(0, 2π, 16)
    Z = [mean(value_of(make_state(target_alt_km=900, rel_alt_km=ra, sep_deg=sp, target_phase=φ)) for φ in φsamp)
         for ra in alt_ax, sp in sep_ax]
    p = heatmap(sep_ax, alt_ax, Z, c=:magma, xlabel="phase separation  Δν  (deg)",
                ylabel="relative altitude  Δa  (km)", title="Observer value over relative orbit geometry",
                colorbar_title="  V", size=(820, 640))
    # training support box
    plot!(p, [10,120,120,10,10], [-300,-300,300,300,-300], c=:cyan, lw=2, ls=:dash, label="train support")
    plot!(p, [-10,-120,-120,-10,-10], [-300,-300,300,300,-300], c=:cyan, lw=2, ls=:dash, label="")
    scatter!(p, [0], [0], c=:white, ms=6, markershape=:star5, label="co-located")
    savefig(joinpath(figdir, "idea2_relative_geometry_map.png"))
    println("wrote idea2_relative_geometry_map.png")
end

# ===========================================================================
# IDEA 3 — Adversarial suppression. Compare the learned EQUILIBRIUM value (target
# is free to evade) against the passive-target opportunity (nobody maneuvers, so
# the observer just racks up detections). The gap = detection opportunity the
# target denies by maneuvering — a game-theoretic feature no myopic sensor model
# would capture.
# ===========================================================================
begin
    sep_ax = LinRange(-140, 140, 60)
    alt_ax = LinRange(-500, 500, 60)
    φ0 = deg2rad(60)  # a sunlit orbital position
    states = [make_state(target_alt_km=900, rel_alt_km=ra, sep_deg=sp, target_phase=φ0)
              for ra in alt_ax, sp in sep_ax]
    Vlearn = reshape(values_of(vec(states)), size(states))
    Vpassive = reshape([noburn_return(s; depth=50) for s in vec(states)], size(states))
    denied = Vpassive .- Vlearn        # >0 : equilibrium value below passive opportunity
    p1 = heatmap(sep_ax, alt_ax, Vlearn, c=:magma, clims=(0, maximum(Vlearn)),
                 title="learned equilibrium V", xlabel="Δν (deg)", ylabel="Δa (km)")
    p2 = heatmap(sep_ax, alt_ax, Vpassive, c=:magma, clims=(0, maximum(Vpassive)),
                 title="passive-target opportunity", xlabel="Δν (deg)")
    p3 = heatmap(sep_ax, alt_ax, denied, c=:thermal,
                 title="opportunity denied by evasion", xlabel="Δν (deg)")
    plot(p1, p2, p3, layout=(1,3), size=(1400, 460), bottom_margin=6Plots.mm, left_margin=5Plots.mm)
    savefig(joinpath(figdir, "idea3_adversarial_suppression.png"))
    println("wrote idea3_adversarial_suppression.png")
end

# ===========================================================================
# IDEA 4 — Gradient saliency: which physical features drive the value?
# mean |dV/dx_i| over the in-distribution state bank.
# ===========================================================================
begin
    Random.seed!(1)
    s0 = core_initialstate_distribution(game)
    X = reduce(hcat, [feat(rand(s0)) for _ in 1:2000])
    critic_val(x) = only(AZ.value(oracle, x))
    g = similar(X)
    for j in 1:size(X, 2)
        g[:, j] = Flux.gradient(x -> critic_val(x), X[:, j])[1]
    end
    # Weight raw gradients by each feature's in-distribution std, giving the
    # expected value change from a realistic 1σ move in that feature (scale-fair).
    fstd = vec(std(X; dims=2))
    sal = vec(mean(abs, g; dims=2)) .* fstd
    labels = ["obs x","obs y","obs vx","obs vy","tar x","tar y","tar vx","tar vy",
              "rel x","rel y","rel vx","rel vy","sinθ","cosθ","visible","snr_norm"]
    order = sortperm(sal)
    p = bar(sal[order], orientation=:h, yticks=(1:16, labels[order]), xlims=(0, maximum(sal)*1.05),
            xlabel="mean |∂V/∂feature| × σ(feature)", title="Value sensitivity (1σ-scaled saliency)",
            size=(720, 620), left_margin=6Plots.mm, legend=false, c=:steelblue)
    savefig(joinpath(figdir, "idea4_saliency.png"))
    println("wrote idea4_saliency.png")
    for i in reverse(order)
        println("  ", rpad(labels[i], 10), round(sal[i]; digits=3))
    end
end

println("DONE")
