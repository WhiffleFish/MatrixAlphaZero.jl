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

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
function make_state(; target_alt_km, rel_alt_km, sep_deg, target_phase)
    r_t = R_EARTH + target_alt_km * 1e3
    r_o = R_EARTH + (target_alt_km + rel_alt_km) * 1e3
    φ_o = mod2pi(target_phase + deg2rad(sep_deg))
    SNRGame.SDAState2D(SNRGame.sOSCtoCART2D([r_o,0.0,0.0,φ_o]),
                       SNRGame.sOSCtoCART2D([r_t,0.0,0.0,target_phase]), game.epc0, false)
end

sep_ax = LinRange(-140, 140, 80)
alt_ax = LinRange(-500, 500, 80)
const BURNS = [-100.0, 0.0, 100.0]

# batch feature matrix for a whole (alt,sep) grid at fixed target phase
function grid_batch(target_phase; target_alt_km=900.0)
    states = [make_state(; target_alt_km, rel_alt_km=ra, sep_deg=sp, target_phase)
              for ra in alt_ax, sp in sep_ax]
    X = reduce(hcat, feat.(vec(states)))
    return states, X
end

# ===========================================================================
# (A) GIF — equilibrium value over relative geometry, as the encounter's
# orbital position (and thus Sun geometry) sweeps around the orbit.
# ===========================================================================
begin
    φs = LinRange(0, 2π, 48)
    Zs = map(φs) do φ
        _, X = grid_batch(φ)
        reshape(vec(Float64.(AZ.value(oracle, X))), length(alt_ax), length(sep_ax))
    end
    vlo, vhi = minimum(minimum.(Zs)), maximum(maximum.(Zs))
    anim = @animate for (φ, Z) in zip(φs, Zs)
        heatmap(sep_ax, alt_ax, Z, c=:magma, clims=(vlo, vhi),
                xlabel="phase separation Δν (deg)", ylabel="relative altitude Δa (km)",
                title="observer value  |  orbital position = $(round(Int, rad2deg(φ)))° from Sun",
                colorbar_title="  V", size=(760, 600))
    end
    gif(anim, joinpath(figdir, "value_sun_sweep.gif"), fps=10)
    println("wrote value_sun_sweep.gif")
end

# ===========================================================================
# (B) POLICY MAP — equilibrium strategy (expected burn) for each player over
# relative geometry, at a fixed sunlit orbital position. Shows the competitive
# push/pull directly: target flees, observer closes.
# ===========================================================================
begin
    φ0 = deg2rad(60)
    states, X = grid_batch(φ0)
    V = reshape(vec(Float64.(AZ.value(oracle, X))), length(alt_ax), length(sep_ax))
    p1probs, p2probs = AZ.strategy(oracle, X)          # each (3, N)
    Edv_obs = reshape(vec(BURNS' * Float64.(p1probs)), length(alt_ax), length(sep_ax))
    Edv_tar = reshape(vec(BURNS' * Float64.(p2probs)), length(alt_ax), length(sep_ax))
    amp = max(maximum(abs, Edv_obs), maximum(abs, Edv_tar))

    pv = heatmap(sep_ax, alt_ax, V, c=:magma, title="value V",
                 xlabel="Δν (deg)", ylabel="Δa (km)")
    po = heatmap(sep_ax, alt_ax, Edv_obs, c=:balance, clims=(-amp, amp),
                 title="observer  E[Δv]  (m/s)", xlabel="Δν (deg)")
    pt = heatmap(sep_ax, alt_ax, Edv_tar, c=:balance, clims=(-amp, amp),
                 title="target  E[Δv]  (m/s)", xlabel="Δν (deg)")
    plot(pv, po, pt, layout=(1,3), size=(1450, 470), bottom_margin=6Plots.mm, left_margin=6Plots.mm)
    savefig(joinpath(figdir, "policy_map.png"))
    println("wrote policy_map.png  |  obs burn range ", round.(extrema(Edv_obs)),
            "  tar burn range ", round.(extrema(Edv_tar)))
end
println("DONE")
