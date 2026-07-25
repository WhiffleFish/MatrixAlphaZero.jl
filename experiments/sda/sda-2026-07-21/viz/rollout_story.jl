using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using MarkovGames, MatrixAlphaZero, Flux, JLD2, POMDPTools, Distributions
using SDAGames.SNRGame, SDAGames.SatelliteDynamics, ExperimentTools
using LinearAlgebra, Random, Statistics, Plots
const AZ = MatrixAlphaZero
const Tools = ExperimentTools
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

const expdir = joinpath(@__DIR__, "..")
include(joinpath(expdir, "initial_state.jl"))
const figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
oracle = AZ.load_oracle(joinpath(expdir, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
Flux.loadmodel!(oracle, JLD2.load(joinpath(expdir,
    "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"])
const μ = game.μ

search = AZ.MCTSSearch(; oracle, tree_queries=100, max_depth=5, max_time=Inf,
    search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
    value_target=:search, ϵ=_->0.1, prior_scale=0.0)
planner = AZ.AlphaZeroPlanner(game, search)

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
value_of(s) = only(Float64.(AZ.value(oracle, feat(s))))
alt_km(x) = (norm(x[1:2]) - R_EARTH) / 1e3
function osc(x)
    r = x[1:2]; v = x[3:4]; rn = norm(r); vn = norm(v)
    return (; a = 1/(2/rn - vn^2/μ), ν = atan(r[2], r[1]))
end
wrap(θ) = mod(θ + π, 2π) - π

# feature groups: collapse 16 raw features into interpretable physics groups
const GROUPS = [
    "observer orbit"    => 1:4,
    "target orbit"      => 5:8,
    "relative geometry" => 9:12,
    "illumination"      => 13:14,
    "detectability"     => 15:16,
]

function rollout(s; steps=60, seed=0)
    rng = MersenneTwister(seed)
    T = NamedTuple[]
    for t in 1:steps
        MarkovGames.isterminal(game, s) && break
        x = feat(s)
        σ = MarkovGames.behavior(planner, s)
        d1, d2 = σ.dists[1], σ.dists[2]
        p1 = Float64.(collect(d1.probs)); p2 = Float64.(collect(d2.probs))
        v1 = [only(v) for v in d1.vals];  v2 = [only(v) for v in d2.vals]
        a1 = v1[rand(rng, Categorical(p1 ./ sum(p1)))]
        a2 = v2[rand(rng, Categorical(p2 ./ sum(p2)))]
        oo = osc(s.observer); ot = osc(s.target)
        sp = rand(rng, MarkovGames.transition(game, s, (a1, a2)))
        r = MarkovGames.reward(game, s, (a1, a2), sp)[1]
        grad = Flux.gradient(z -> only(AZ.value(oracle, z)), x)[1]
        push!(T, (; t, V=value_of(s), snr=Float64(x[16]), visible=x[15] > 0.5,
            reward=r, detected = r >= 1.0, a1, a2,
            Edv1 = v1'*p1, Edv2 = v2'*p2, grad = Float64.(grad), x = Float64.(x),
            range_km = norm(s.observer[1:2] .- s.target[1:2])/1e3,
            obs_alt = alt_km(s.observer), tar_alt = alt_km(s.target),
            dnu = rad2deg(wrap(oo.ν - ot.ν)),
            obs = collect(s.observer[1:2]), tar = collect(s.target[1:2])))
        s = sp
    end
    return T
end

Random.seed!(7)
s0 = core_initialstate_distribution(game)
best = nothing
for k in 1:12
    Random.seed!(100 + k)
    T = rollout(rand(s0); steps=60, seed=k)
    score = count(x -> x.detected, T) * count(x -> abs(x.a2) > 0, T)
    if isnothing(best) || score > best.score
        global best = (; score, T, k)
    end
end
T = best.T
println("candidate $(best.k): $(length(T)) steps, $(count(x->x.detected,T)) detections")

ts = getfield.(T, :t); V = getfield.(T, :V); snr = getfield.(T, :snr)
det = getfield.(T, :detected); vis = getfield.(T, :visible)
Edv1 = getfield.(T, :Edv1); Edv2 = getfield.(T, :Edv2)
oalt = getfield.(T, :obs_alt); talt = getfield.(T, :tar_alt)
detspan = ts[det]
shade!(p) = for t in detspan; vspan!(p, [t-0.5,t+0.5], c=:gold, alpha=0.28, label=""); end

# in-distribution feature scale, for scale-fair saliency
Random.seed!(1)
Xbank = reduce(hcat, [feat(rand(s0)) for _ in 1:2000])
fstd = vec(std(Float64.(Xbank); dims=2))

# ===========================================================================
# FIGURE A — the rollout as a story: pursuit in altitude, value/SNR, burns,
# and which feature group drives the value at each moment.
# ===========================================================================
begin
    pA = plot(ts, talt, lw=3, c=:darkorange, label="target", ylabel="altitude (km)",
              title="equilibrium rollout: target flees outward, observer pursues", yscale=:log10)
    plot!(pA, ts, oalt, lw=3, c=:navy, label="observer")
    shade!(pA); plot!(pA, ts, talt, lw=3, c=:darkorange, label=""); plot!(pA, ts, oalt, lw=3, c=:navy, label="")

    pB = plot(ts, V, lw=3, c=:black, ylabel="value  V", label="V")
    shade!(pB); plot!(pB, ts, V, lw=3, c=:black, label="")
    pB2 = twinx(pB)
    plot!(pB2, ts, snr, lw=2.5, c=:crimson, ls=:dash, yscale=:log10,
          ylabel="SNR / threshold", label="")
    hline!(pB2, [1.0], ls=:dot, c=:gray, label="")

    pC = plot(ts, Edv2, lw=3, c=:darkorange, label="target E[Δv]", ylabel="E[Δv] (m/s)")
    plot!(pC, ts, Edv1, lw=3, c=:navy, label="observer E[Δv]")
    hline!(pC, [0], ls=:dot, c=:gray, label=""); shade!(pC)
    plot!(pC, ts, Edv2, lw=3, c=:darkorange, label=""); plot!(pC, ts, Edv1, lw=3, c=:navy, label="")

    # per-step grouped saliency: |∂V/∂x_i| * σ_i summed within each group,
    # normalized per step to show *which* group dominates at each moment.
    S = zeros(length(GROUPS), length(T))
    for (j, step) in enumerate(T)
        contrib = abs.(step.grad) .* fstd
        for (i, (_, idx)) in enumerate(GROUPS)
            S[i, j] = sum(contrib[idx])
        end
        S[:, j] ./= sum(S[:, j])
    end
    pD = heatmap(ts, 1:length(GROUPS), S, c=:viridis,
                 yticks=(1:length(GROUPS), first.(GROUPS)), xlabel="step",
                 colorbar_title="  share of |∂V|", title="which features drive the value")

    plot(pA, pB, pC, pD, layout=(4,1), size=(1050, 1250),
         left_margin=14Plots.mm, right_margin=14Plots.mm, bottom_margin=3Plots.mm)
    savefig(joinpath(figdir, "rollout_story.png"))
    println("wrote rollout_story.png")
    for (i,(name,_)) in enumerate(GROUPS)
        println("  mean share ", rpad(name,20), round(mean(S[i,:]); digits=3))
    end
end

# ===========================================================================
# FIGURE B — trajectories: full escape view + zoom on the detection phase.
# ===========================================================================
begin
    ox = [x.obs[1] for x in T]; oy = [x.obs[2] for x in T]
    tx = [x.tar[1] for x in T]; ty = [x.tar[2] for x in T]
    θ = LinRange(0, 2π, 200)
    di = findall(det)
    # big evasive burns only, so the markers mean something
    bi = findall(x -> x.Edv2 > 60, T)

    function traj(lim, ttl; showlegend=false)
        p = plot(aspect_ratio=1, xlabel="x (m)", ylabel="y (m)", title=ttl,
                 xlims=(-lim,lim), ylims=(-lim,lim), legend=showlegend ? :topright : false)
        plot!(p, R_EARTH.*cos.(θ), R_EARTH.*sin.(θ), c=:gray70, lw=0,
              fill=(0,:gray85), label="Earth")
        plot!(p, tx, ty, c=:darkorange, lw=2.5, label="target path")
        plot!(p, ox, oy, c=:navy, lw=1.5, alpha=0.55, label="observer path")
        scatter!(p, ox, oy, zcolor=V, c=:magma, ms=4.5, msw=0, label="", colorbar=false)
        isempty(bi) || scatter!(p, tx[bi], ty[bi], mc=:red, ms=4.5, markershape=:diamond,
                                msw=0, label="target evasive burn")
        isempty(di) || scatter!(p, ox[di], oy[di], mc=:gold, ms=9, markershape=:star5,
                                msw=0.6, msc=:black, label="detection")
        return p
    end
    p1 = traj(2.6e7, "full episode: target escapes outward", showlegend=true)
    plot!(p1, [0, 2.2e7], [0, 0], arrow=true, c=:goldenrod, lw=2.5, label="to Sun")
    p2 = traj(1.1e7, "zoom: early detection phase")
    plot(p1, p2, layout=(1,2), size=(1500, 720), bottom_margin=8Plots.mm, left_margin=6Plots.mm)
    savefig(joinpath(figdir, "rollout_trajectory.png"))
    println("wrote rollout_trajectory.png")
end

# ===========================================================================
# FIGURE C — saliency, clearly: grouped bars + where each group dominates,
# as a function of the competitive situation (range).
# ===========================================================================
begin
    Random.seed!(3)
    states = [rand(s0) for _ in 1:1500]
    X = reduce(hcat, feat.(states))
    G = similar(X, Float64)
    for j in 1:size(X,2)
        G[:,j] = Flux.gradient(z -> only(AZ.value(oracle, z)), X[:,j])[1]
    end
    contrib = abs.(G) .* fstd                     # (16, N)
    gshare = reduce(vcat, [sum(contrib[idx,:], dims=1) for (_,idx) in GROUPS])  # (5, N)
    gshare ./= sum(gshare, dims=1)

    mshare = vec(mean(gshare, dims=2))
    pbar = bar(1:length(GROUPS), mshare, orientation=:h, bar_width=0.62,
               yticks=(1:length(GROUPS), first.(GROUPS)), c=:steelblue, legend=false,
               xlims=(0, maximum(mshare)*1.18), ylims=(0.4, length(GROUPS)+0.6),
               xlabel="mean share of value sensitivity", title="what the value function uses")
    for (i, m) in enumerate(mshare)
        annotate!(pbar, m + maximum(mshare)*0.03, i, text(string(round(m; digits=3)), 9, :left))
    end

    # conditioned on range: which group dominates when close vs far?
    rngs = [norm(s.observer[1:2] .- s.target[1:2])/1e3 for s in states]
    edges = 10 .^ range(log10(max(minimum(rngs),100)), log10(maximum(rngs)), length=9)
    ctrs = sqrt.(edges[1:end-1] .* edges[2:end])
    M = fill(NaN, length(GROUPS), length(ctrs))
    for b in 1:length(ctrs)
        sel = findall(r -> edges[b] <= r < edges[b+1], rngs)
        length(sel) < 12 && continue
        M[:, b] = vec(mean(gshare[:, sel], dims=2))
    end
    cols = [:navy, :darkorange, :seagreen, :goldenrod, :crimson]
    keep = findall(b -> !isnan(M[1,b]), 1:length(ctrs))
    pline = areaplot(ctrs[keep], permutedims(M[:, keep]), seriescolor=permutedims(cols),
                     label=permutedims(first.(GROUPS)), fillalpha=0.85, lw=0,
                     xscale=:log10, xlabel="observer–target range (km)",
                     ylabel="share of value sensitivity", ylims=(0,1),
                     title="feature importance vs engagement range", legend=:outerright)
    plot(pbar, pline, layout=(1,2), size=(1500, 520),
         left_margin=12Plots.mm, bottom_margin=8Plots.mm)
    savefig(joinpath(figdir, "saliency_grouped.png"))
    println("wrote saliency_grouped.png")
end
println("DONE")
