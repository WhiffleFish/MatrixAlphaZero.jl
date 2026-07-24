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
const μ = game.μ

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
value_of(s) = only(Float64.(AZ.value(oracle, feat(s))))

# --- osculating elements from a planar Cartesian state [x,y,vx,vy] ------------
function osc(x)
    r = x[1:2]; v = x[3:4]
    rn = norm(r); vn = norm(v)
    a = 1 / (2/rn - vn^2/μ)                 # vis-viva semi-major axis
    evec = ((vn^2 - μ/rn) .* r .- dot(r, v) .* v) ./ μ
    e = norm(evec)
    ν = atan(r[2], r[1])                    # true longitude (planar)
    return (; a, e, ν)
end
alt_km(x) = (osc(x).a - R_EARTH) / 1e3
wrap(θ) = mod(θ + π, 2π) - π

function make_state(; target_alt_km, rel_alt_km, sep_deg, target_phase)
    r_t = R_EARTH + target_alt_km * 1e3
    r_o = R_EARTH + (target_alt_km + rel_alt_km) * 1e3
    φ_o = mod2pi(target_phase + deg2rad(sep_deg))
    SNRGame.SDAState2D(SNRGame.sOSCtoCART2D([r_o,0.0,0.0,φ_o]),
                       SNRGame.sOSCtoCART2D([r_t,0.0,0.0,target_phase]), game.epc0, false)
end

# ===========================================================================
# Q2 — What states does TRAINING actually visit? Roll out under maneuvering
# (random burns, matching the sim_depth horizon) and record where the pair goes
# in (Δν, Δa) AND how much eccentricity the burns induce (bears on OSC encoding).
# ===========================================================================
begin
    Random.seed!(2)
    s0 = core_initialstate_distribution(game)
    A1, A2 = MarkovGames.actions(game)
    dΔν = Float64[]; dΔa = Float64[]; ecc = Float64[]
    for _ in 1:400
        s = rand(s0)
        for t in 1:50
            MarkovGames.isterminal(game, s) && break
            oo = osc(s.observer); ot = osc(s.target)
            push!(dΔa, (oo.a - ot.a)/1e3)
            push!(dΔν, rad2deg(wrap(oo.ν - ot.ν)))
            push!(ecc, oo.e); push!(ecc, ot.e)
            a = (rand(A1), rand(A2))
            s = rand(MarkovGames.transition(game, s, a))
        end
    end
    # initial-state box for reference
    p1 = scatter(dΔν, dΔa, ms=1.4, mc=:steelblue, ma=0.25, msw=0,
                 xlabel="phase separation Δν (deg)", ylabel="relative altitude Δa (km)",
                 title="states reached under maneuvering (50-step rollouts)",
                 xlims=(-180,180), ylims=(-2000,2000))
    plot!(p1, [10,120,120,10,10],[-300,-300,300,300,-300], c=:crimson, lw=2, label="initial-state box")
    plot!(p1, [-10,-120,-120,-10,-10],[-300,-300,300,300,-300], c=:crimson, lw=2, label="")
    p2 = histogram(ecc, bins=60, normalize=:pdf, c=:seagreen, lw=0,
                   xlabel="osculating eccentricity e", ylabel="density",
                   title="eccentricity induced by burns (e=0 at t=0)")
    plot(p1, p2, layout=(1,2), size=(1300,520), bottom_margin=6Plots.mm, left_margin=6Plots.mm)
    savefig(joinpath(figdir, "q2_visited_support.png"))
    println("wrote q2_visited_support.png  |  Δa range reached: ",
            round(minimum(dΔa)), "..", round(maximum(dΔa)), " km ; ",
            "e max=", round(maximum(ecc); digits=3), " median=", round(median(ecc); digits=3))
end

# ===========================================================================
# Q4 — Saliency as a FIELD, not a bar chart. Over the orbit plane, plot the
# value heatmap and the gradient of V w.r.t. observer position: the direction
# in which moving the observer most increases its value (the "pursuit field").
# ===========================================================================
begin
    target_alt = 900.0; target_phase = deg2rad(60)
    s_t = SNRGame.sOSCtoCART2D([R_EARTH+target_alt*1e3, 0.0, 0.0, target_phase])
    # background heatmap: circular-orbit observers over altitude x phase
    alt_ax = LinRange(400, 1500, 120); ph_ax = LinRange(0, 2π, 200)
    xs = Float64[]; ys = Float64[]; vs = Float64[]
    for al in alt_ax, ph in ph_ax
        so = SNRGame.sOSCtoCART2D([R_EARTH+al*1e3, 0.0, 0.0, ph])
        s = SNRGame.SDAState2D(so, s_t, game.epc0, false)
        push!(xs, so[1]); push!(ys, so[2]); push!(vs, value_of(s))
    end
    p = scatter(xs, ys, zcolor=vs, c=:magma, ms=2.2, msw=0, ma=0.9, label="",
                colorbar_title="  V", aspect_ratio=1, size=(820,780),
                xlabel="x (m)", ylabel="y (m)", title="value + pursuit gradient field")

    # gradient quiver on a coarser grid
    critic_val(x) = only(AZ.value(oracle, x))
    qx=Float64[]; qy=Float64[]; ux=Float64[]; uy=Float64[]
    for al in LinRange(450,1450,11), ph in LinRange(0,2π,28)
        so = SNRGame.sOSCtoCART2D([R_EARTH+al*1e3, 0.0, 0.0, ph])
        s = SNRGame.SDAState2D(so, s_t, game.epc0, false)
        x = feat(s)
        g = Flux.gradient(critic_val, x)[1]
        dx, dy = g[1], g[2]        # ∂V/∂(obs x, obs y)  (features 1,2)
        n = hypot(dx,dy); n < 1e-9 && continue
        L = 1.2e6
        push!(qx, so[1]); push!(qy, so[2]); push!(ux, dx/n*L); push!(uy, dy/n*L)
    end
    quiver!(p, qx, qy, quiver=(ux,uy), c=:cyan, lw=1.2)
    scatter!(p, [s_t[1]], [s_t[2]], c=:lime, ms=8, markershape=:star5, label="target")
    savefig(joinpath(figdir, "q4_pursuit_field.png"))
    println("wrote q4_pursuit_field.png")
end
println("DONE")
