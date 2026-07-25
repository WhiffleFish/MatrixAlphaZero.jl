#=
Does the Cartesian-trained critic ALREADY encode orbital elements internally?

Linear-probe each hidden layer of the frozen critic for equinoctial quantities
(a_obs, a_tar, Δa, e_obs, e_tar, sin/cos Δλ). High R² => the net reconstructed
element structure on its own, so an element encoding mostly saves it work it has
already done. Low R² (especially for Δa/e) => the encoding would supply something
the net never learned.

Baseline: the same ridge regression run on the RAW Cartesian input features, which
says how much of each element is linearly available before any network processing.
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
critic = oracle.critic

cart_feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)

function equinoctial(x)
    r = @view x[1:2]; v = @view x[3:4]
    rn = norm(r); vn = norm(v)
    a = 1 / (2/rn - vn^2/μ)
    evec = ((vn^2 - μ/rn) .* r .- dot(r, v) .* v) ./ μ
    k, h = evec[1], evec[2]
    e = hypot(h, k)
    ν = atan(r[2], r[1]); ϖ = atan(h, k); f = ν - ϖ
    E = atan(sqrt(max(1-e^2,0.0))*sin(f), e + cos(f))
    return (; a, h, k, e, λ = ϖ + (E - e*sin(E)))
end

# dataset over visited support
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
    S
end

Random.seed!(0)
S = build_dataset(800, 50)
X = reduce(hcat, cart_feat.(S))
println("probe dataset: ", length(S), " states")

# targets
eo = equinoctial.(getfield.(S, :observer))
et = equinoctial.(getfield.(S, :target))
Δλ = [o.λ - t.λ for (o,t) in zip(eo,et)]
targets = Dict(
    "a_obs"   => [(o.a - R_EARTH)/1e6 for o in eo],
    "a_tar"   => [(t.a - R_EARTH)/1e6 for t in et],
    "Δa"      => [(o.a - t.a)/1e6 for (o,t) in zip(eo,et)],
    "e_obs"   => [o.e for o in eo],
    "e_tar"   => [t.e for t in et],
    "sin Δλ"  => sin.(Δλ),
    "cos Δλ"  => cos.(Δλ),
)
order = ["a_obs","a_tar","Δa","e_obs","e_tar","sin Δλ","cos Δλ"]

# hidden activations: critic is Chain(Dense,tanh) x3 + Dense(->1)
acts = Dict{String,Matrix{Float32}}()
acts["input (raw)"] = X
let act = X
    for (i, layer) in enumerate(critic.layers[1:end-1])
        act = layer(act)
        acts["hidden $i"] = copy(act)
    end
end
layer_names = ["input (raw)", "hidden 1", "hidden 2", "hidden 3"]

"ridge-regression R² with a held-out split"
function probe_r2(A, y; λreg=1e-3)
    n = size(A, 2)
    idx = randperm(n); ntr = div(3n, 4)
    tr, te = idx[1:ntr], idx[ntr+1:end]
    Atr = vcat(Float64.(A[:, tr]), ones(1, length(tr)))
    Ate = vcat(Float64.(A[:, te]), ones(1, length(te)))
    ytr, yte = y[tr], y[te]
    m, s = mean(Atr; dims=2), std(Atr; dims=2) .+ 1e-9
    s[end] = 1.0; m[end] = 0.0
    Atr = (Atr .- m) ./ s; Ate = (Ate .- m) ./ s
    d = size(Atr, 1)
    w = (Atr*Atr' + λreg*I(d)) \ (Atr*ytr)
    pred = Ate' * w
    return 1 - sum((pred .- yte).^2) / sum((yte .- mean(yte)).^2)
end

Random.seed!(1)
R2 = [probe_r2(acts[ln], targets[t]) for t in order, ln in layer_names]
println("\nlinear decodability (R²):")
@printf("%-9s", "target"); foreach(ln -> @printf("%14s", ln), layer_names); println()
for (i, t) in enumerate(order)
    @printf("%-9s", t); foreach(j -> @printf("%14.3f", R2[i,j]), 1:length(layer_names)); println()
end

p = heatmap(layer_names, order, clamp.(R2, 0, 1), c=:viridis, clims=(0,1),
            title="Are orbital elements linearly decodable from the critic?",
            xlabel="", ylabel="equinoctial quantity", size=(820, 520),
            colorbar_title="  R²", left_margin=6Plots.mm, bottom_margin=5Plots.mm)
for i in 1:size(R2,1), j in 1:size(R2,2)
    annotate!(p, j-0.5+0.5, i-0.5+0.5, text(string(round(clamp(R2[i,j],0,1); digits=2)), 9,
              clamp(R2[i,j],0,1) > 0.55 ? :black : :white))
end
savefig(joinpath(figdir, "element_probe.png"))
println("\nwrote element_probe.png")
