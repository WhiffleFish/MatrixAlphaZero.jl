#=
Encoding A/B: is the learned value function easier to represent in equinoctial
element space than in the current Cartesian encoding?

Method: freeze the trained oracle's critic as ground truth, then DISTILL it into
identical MLPs that differ only in input encoding. Same width/depth/optimizer/
epochs, z-scored inputs, several train-set sizes and seeds. Test error vs data
size is a direct read on which representation makes the target function easier.

States come from random-burn rollouts (the actual visited support, where e > 0),
not just the initial-state distribution.
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
using MarkovGames, MatrixAlphaZero, Flux, JLD2, POMDPTools, Distributions
using SDAGames.SNRGame, SDAGames.SatelliteDynamics
using LinearAlgebra, Random, Statistics, Printf, Plots
const AZ = MatrixAlphaZero
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

const expdir = joinpath(@__DIR__, "..")
const figdir = joinpath(@__DIR__, "figs"); mkpath(figdir)
include(joinpath(expdir, "initial_state.jl"))

game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
oracle = AZ.load_oracle(joinpath(expdir, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
Flux.loadmodel!(oracle, JLD2.load(joinpath(expdir,
    "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"])
const μ = game.μ

# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------
cart_feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)

"Planar non-singular equinoctial elements: a, h=e sin ϖ, k=e cos ϖ, mean longitude λ."
function equinoctial(x)
    r = @view x[1:2]; v = @view x[3:4]
    rn = norm(r); vn = norm(v)
    a = 1 / (2/rn - vn^2/μ)
    evec = ((vn^2 - μ/rn) .* r .- dot(r, v) .* v) ./ μ
    k, h = evec[1], evec[2]              # e vector components == (k, h) in-plane
    e = hypot(h, k)
    # eccentric/mean anomaly -> mean longitude λ = ϖ + M (well-defined as e->0)
    ν = atan(r[2], r[1])                 # true longitude
    ϖ = atan(h, k)                       # longitude of periapsis
    f = ν - ϖ                            # true anomaly
    E = atan(sqrt(max(1 - e^2, 0.0)) * sin(f), e + cos(f))
    M = E - e * sin(E)
    λ = ϖ + M
    return (; a, h, k, λ)
end

"""
Equinoctial encoding. Per satellite: a (scaled), h, k, sin/cos of mean longitude.
Plus RELATIVE elements (the competitive coordinates) and the same illumination /
detectability features the Cartesian encoding uses (these are photometric, not
orbital, so both encodings get them — the A/B isolates the ORBITAL representation).
"""
function eqx_feat(s)
    o = equinoctial(s.observer); t = equinoctial(s.target)
    c = cart_feat(s)                       # reuse features 13:16
    Δλ = o.λ - t.λ
    Float32[
        (o.a - R_EARTH)/1e7, o.h, o.k, sin(o.λ), cos(o.λ),
        (t.a - R_EARTH)/1e7, t.h, t.k, sin(t.λ), cos(t.λ),
        (o.a - t.a)/1e6,                   # Δa (km-ish scale)
        o.h - t.h, o.k - t.k,
        sin(Δλ), cos(Δλ),                  # relative phase
        c[13], c[14], c[15], c[16],        # sinθ, cosθ, visible, snr_norm
    ]
end

# ---------------------------------------------------------------------------
# Dataset: visited support via random-burn rollouts
# ---------------------------------------------------------------------------
function build_dataset(n_traj, depth)
    s0 = core_initialstate_distribution(game)
    A1, A2 = MarkovGames.actions(game)
    S = SNRGame.SDAState2D[]
    for _ in 1:n_traj
        s = rand(s0)
        for _ in 1:depth
            MarkovGames.isterminal(game, s) && break
            push!(S, s)
            s = rand(MarkovGames.transition(game, s, (rand(A1), rand(A2))))
        end
    end
    return S
end

Random.seed!(0)
S = build_dataset(1200, 50)
println("dataset: ", length(S), " states")

Xc = reduce(hcat, cart_feat.(S))
Xe = reduce(hcat, eqx_feat.(S))
y  = vec(Float64.(AZ.value(oracle, Xc)))       # frozen oracle = ground truth
println("target V: mean=", round(mean(y);digits=2), " std=", round(std(y);digits=2))

# shuffle + split
perm = randperm(length(y))
Xc, Xe, y = Xc[:, perm], Xe[:, perm], y[perm]
ntest = 8000
Xc_te, Xe_te, y_te = Xc[:, 1:ntest], Xe[:, 1:ntest], y[1:ntest]
Xc_tr, Xe_tr, y_tr = Xc[:, ntest+1:end], Xe[:, ntest+1:end], y[ntest+1:end]
println("train pool=", size(Xc_tr,2), "  test=", ntest)

zstats(X) = (mean(X; dims=2), std(X; dims=2) .+ 1f-6)
apply_z(X, (m, s)) = (X .- m) ./ s

function fit_probe(Xtr, ytr, Xte, yte; width=64, epochs=400, seed=0, lr=3f-4)
    Random.seed!(seed)
    st = zstats(Xtr)
    A, B = apply_z(Xtr, st), apply_z(Xte, st)
    ym, ys = mean(ytr), std(ytr)
    a = Float32.(reshape((ytr .- ym) ./ ys, 1, :))
    din = size(Xtr, 1)
    m = Chain(Dense(din=>width, tanh), Dense(width=>width, tanh),
              Dense(width=>width, tanh), Dense(width=>1))
    opt = Flux.setup(Flux.Adam(lr), m)
    n = size(A, 2); bs = 256
    for ep in 1:epochs
        for idx in Iterators.partition(randperm(n), bs)
            xb, yb = A[:, idx], a[:, idx]
            g = Flux.gradient(mm -> Flux.mse(mm(xb), yb), m)[1]
            Flux.update!(opt, m, g)
        end
    end
    pred = vec(Float64.(m(B))) .* ys .+ ym
    rmse = sqrt(mean((pred .- yte).^2))
    return rmse
end

sizes = [500, 2000, 8000, 32000]
seeds = [0, 1, 2]
res = Dict("cartesian"=>Float64[], "equinoctial"=>Float64[])
sd  = Dict("cartesian"=>Float64[], "equinoctial"=>Float64[])
for nt in sizes
    nt > size(Xc_tr,2) && continue
    for (name, Xtr, Xte) in (("cartesian", Xc_tr, Xc_te), ("equinoctial", Xe_tr, Xe_te))
        rs = [fit_probe(Xtr[:,1:nt], y_tr[1:nt], Xte, y_te; seed=sd_) for sd_ in seeds]
        push!(res[name], mean(rs)); push!(sd[name], std(rs))
        @printf("n=%6d  %-12s  test RMSE = %6.3f ± %.3f\n", nt, name, mean(rs), std(rs))
    end
end

used = filter(n -> n <= size(Xc_tr,2), sizes)
p = plot(used, res["cartesian"], yerror=sd["cartesian"], lw=3, m=:circle, ms=6,
         label="Cartesian (current)", xscale=:log10, xlabel="distillation training states",
         ylabel="test RMSE  (value units)", title="Which encoding represents V more easily?",
         legend=:topright, size=(760,560), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
plot!(p, used, res["equinoctial"], yerror=sd["equinoctial"], lw=3, m=:square, ms=6,
      label="Equinoctial elements")
hline!(p, [std(y_te)], ls=:dash, c=:gray, label="predict-the-mean")
savefig(joinpath(figdir, "encoding_ab.png"))
println("wrote encoding_ab.png")
