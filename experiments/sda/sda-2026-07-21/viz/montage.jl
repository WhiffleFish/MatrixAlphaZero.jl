using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
using MarkovGames, MatrixAlphaZero, Flux, JLD2, POMDPTools, Distributions
using SDAGames.SNRGame, SDAGames.SatelliteDynamics, LinearAlgebra, Statistics, Plots
const AZ = MatrixAlphaZero
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
const expdir = joinpath(@__DIR__, "..")
include(joinpath(expdir, "initial_state.jl"))

game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
oracle = AZ.load_oracle(joinpath(expdir, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
Flux.loadmodel!(oracle, JLD2.load(joinpath(expdir, "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"])

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
make_state(; ra, sp, φ, tak=900.0) = SNRGame.SDAState2D(
    SNRGame.sOSCtoCART2D([R_EARTH+(tak+ra)*1e3,0.0,0.0,mod2pi(φ+deg2rad(sp))]),
    SNRGame.sOSCtoCART2D([R_EARTH+tak*1e3,0.0,0.0,φ]), game.epc0, false)

sep_ax = LinRange(-140,140,80); alt_ax = LinRange(-500,500,80)
Vmap(φ) = reshape(vec(Float64.(AZ.value(oracle,
    reduce(hcat, [feat(make_state(ra=ra, sp=sp, φ=φ)) for ra in alt_ax, sp in sep_ax] |> vec)))),
    length(alt_ax), length(sep_ax))

φs = deg2rad.([0, 90, 180, 270])
Zs = Vmap.(φs)
vlo, vhi = minimum(minimum.(Zs)), maximum(maximum.(Zs))
ps = map(zip(φs, Zs)) do (φ, Z)
    heatmap(sep_ax, alt_ax, Z, c=:magma, clims=(vlo,vhi), colorbar=false,
            title="$(round(Int,rad2deg(φ)))° from Sun", xlabel="Δν (deg)", ylabel="Δa (km)")
end
plot(ps..., layout=(2,2), size=(1000,820), left_margin=4Plots.mm, bottom_margin=4Plots.mm,
     plot_title="observer value vs orbital position (Sun geometry)")
savefig(joinpath(@__DIR__, "figs", "value_sun_montage.png"))
println("wrote value_sun_montage.png  |  per-frame max V: ", round.(maximum.(Zs); digits=1))
