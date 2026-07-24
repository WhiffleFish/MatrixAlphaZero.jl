using Distributed

# ---------------------------------------------------------------------------
# "Opportunity denied by evasion" surface, animated over orbital position φ.
# For each φ frame and each (Δν, Δa) grid cell we estimate the observer's value
# when it runs search and the target coasts (passive), then subtract the net's
# equilibrium value. All heavy work is parallel search-rollouts across workers.
#
# Run `julia rigorous_denied.jl --help` for options. On a ~30-worker server the
# defaults (40x40x24, eps=4 ≈ 154k rollouts) run in ~20 min; bump --nsep/--nalt/
# --eps for a cleaner figure once you know the per-iteration rate.
# ---------------------------------------------------------------------------
const EXPERIMENTS_ENV = abspath(joinpath(@__DIR__, "..", "..", ".."))
const EXPDIR = abspath(joinpath(@__DIR__, ".."))
const FIGDIR = joinpath(@__DIR__, "figs")

using Pkg
Pkg.activate(EXPERIMENTS_ENV; io=devnull)
using ArgParse

function parse_args_cli()
    s = ArgParseSettings(description="Rigorous opportunity-denied φ-sweep for the SDA value function.")
    @add_arg_table! s begin
        "--nworkers"
            help = "number of worker processes to spin up"
            arg_type = Int
            default = max(1, min(Sys.CPU_THREADS - 2, 8))
        "--nframes"
            help = "φ frames in the sweep (1 => single static frame)"
            arg_type = Int
            default = 24
        "--nsep"
            help = "grid columns (phase separation Δν)"
            arg_type = Int
            default = 40
        "--nalt"
            help = "grid rows (relative altitude Δa)"
            arg_type = Int
            default = 40
        "--eps"
            help = "rollout episodes averaged per grid cell"
            arg_type = Int
            default = 4
    end
    return parse_args(s)
end

const ARGS_CLI = parse_args_cli()
const NWORKERS = max(1, ARGS_CLI["nworkers"])
const NFRAMES  = ARGS_CLI["nframes"]
const NSEP     = ARGS_CLI["nsep"]
const NALT     = ARGS_CLI["nalt"]
const EPS      = ARGS_CLI["eps"]

addprocs(NWORKERS)

@everywhere begin
    using Pkg
    Pkg.activate($(EXPERIMENTS_ENV); io=devnull)
end

@everywhere begin
    using MarkovGames, MatrixAlphaZero, Flux, JLD2, POMDPTools, Distributions
    using SDAGames.SNRGame, SDAGames.SatelliteDynamics, ExperimentTools, Random
    const AZ = MatrixAlphaZero
    const Tools = ExperimentTools

    const _EXPDIR = $(EXPDIR)
    game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
    oracle = AZ.load_oracle(joinpath(_EXPDIR, "oracle_rm_plus_no_transfer_train_mean_leo.jld2"))
    Flux.loadmodel!(oracle, JLD2.load(joinpath(_EXPDIR,
        "models_rm_plus_no_transfer_train_mean_leo", "oracle1221.jld2"))["model_state"])

    _search = AZ.MCTSSearch(; oracle, tree_queries=100, max_depth=5, max_time=Inf,
        search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
        value_target=:search, ϵ=_->0.1, prior_scale=0.0)
    _planner = AZ.AlphaZeroPlanner(game, _search)
    # observer runs search; target coasts (passive)
    observer = Tools.JointPolicy(Tools.SinglePlayerAlphaZeroPolicy(_planner, 1),
                                 Tools.sda_no_burn_heuristic(game, 2))

    # mean observer discounted return from state s, target passive
    function passive_value(s; eps::Int=4, seed::Int=0)
        Random.seed!(seed)
        r = Tools.evaluate_joint_policy(game, observer, eps;
            max_steps=50, initialstates=fill(s, eps),
            accumulators=(), batch_accumulators=(),
            show_progress=false, proc_warn=false, parallel=false)
        return Float64(r.reward[1])
    end
end

using Statistics, Plots, ProgressMeter, JLD2
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
include(joinpath(EXPDIR, "initial_state.jl"))

feat(s) = MarkovGames.convert_s(Vector{Float32}, s, game)
make_state(; ra, sp, φ, tak=900.0) = SNRGame.SDAState2D(
    SNRGame.sOSCtoCART2D([R_EARTH+(tak+ra)*1e3,0.0,0.0,mod2pi(φ+deg2rad(sp))]),
    SNRGame.sOSCtoCART2D([R_EARTH+tak*1e3,0.0,0.0,φ]), game.epc0, false)

# light 3x3 smoothing (edge-replicated) for display only; raw arrays are saved
function smooth3(A)
    m, n = size(A); B = similar(A, Float64)
    for j in 1:n, i in 1:m
        acc = 0.0; c = 0
        for dj in -1:1, di in -1:1
            ii = clamp(i+di, 1, m); jj = clamp(j+dj, 1, n)
            acc += A[ii, jj]; c += 1
        end
        B[i, j] = acc / c
    end
    return B
end

sep_ax = LinRange(-140, 140, NSEP)
alt_ax = LinRange(-500, 500, NALT)
φs = NFRAMES == 1 ? [deg2rad(60)] : collect(LinRange(0, 2π, NFRAMES + 1))[1:NFRAMES]

# grid states for every frame, flattened; grid is frame-independent in shape
frame_states = [[make_state(ra=ra, sp=sp, φ=φ) for ra in alt_ax, sp in sep_ax] for φ in φs]
npix = NSEP * NALT
all_states = reduce(vcat, vec.(frame_states))
ntasks = length(all_states)

# equilibrium value per frame is free from the value net
Veq = [reshape(vec(Float64.(AZ.value(oracle, reduce(hcat, feat.(vec(fs)))))), NALT, NSEP)
       for fs in frame_states]

est_min = ntasks * EPS * 0.223 / NWORKERS / 60
println("=== rigorous opportunity-denied sweep ===")
println("frames=$(NFRAMES)  grid=$(NALT)x$(NSEP)  eps=$(EPS)  workers=$(NWORKERS)")
println("total rollout batches=$(ntasks)  (", ntasks*EPS, " episodes)  rough estimate ~",
        round(est_min; digits=1), " min")

prog = Progress(ntasks; desc="rollouts ", showspeed=true)
tstart = time()
Vpass_flat = progress_pmap(x -> passive_value(x[2]; eps=EPS, seed=x[1]),
                           collect(enumerate(all_states)); progress=prog)
println("finished ", ntasks, " batches in ", round((time()-tstart)/60; digits=2), " min")

Vpass = [reshape(Vpass_flat[(k-1)*npix+1 : k*npix], NALT, NSEP) for k in 1:NFRAMES]
denied = [Vpass[k] .- Veq[k] for k in 1:NFRAMES]

# save raw arrays so any re-plot / re-smooth is free (no re-run)
jldsave(joinpath(FIGDIR, "rigorous_denied_sweep.jld2");
    sep_ax=collect(sep_ax), alt_ax=collect(alt_ax), phis=φs, Veq, Vpass, denied,
    eps=EPS, nframes=NFRAMES)

# global color scales for a stable animation
vhi = maximum(m -> maximum(m), vcat(Veq, Vpass))
dhi = maximum(m -> maximum(abs, m), denied)

function frame_plot(k)
    φdeg = round(Int, rad2deg(φs[k]))
    p1 = heatmap(sep_ax, alt_ax, Veq[k], c=:magma, clims=(0,vhi),
                 title="equilibrium V (net)", xlabel="Δν (deg)", ylabel="Δa (km)")
    p2 = heatmap(sep_ax, alt_ax, smooth3(Vpass[k]), c=:magma, clims=(0,vhi),
                 title="V vs passive target", xlabel="Δν (deg)")
    p3 = heatmap(sep_ax, alt_ax, smooth3(denied[k]), c=:thermal, clims=(0,dhi),
                 title="opportunity denied by evasion", xlabel="Δν (deg)")
    plot(p1, p2, p3, layout=(1,3), size=(1450, 470), bottom_margin=6Plots.mm,
         left_margin=6Plots.mm, plot_title="orbital position $(φdeg)° from Sun")
end

if NFRAMES == 1
    frame_plot(1); savefig(joinpath(FIGDIR, "rigorous_denied_frame.png"))
    println("wrote rigorous_denied_frame.png")
else
    anim = @animate for k in 1:NFRAMES
        frame_plot(k)
    end
    gif(anim, joinpath(FIGDIR, "rigorous_denied_sweep.gif"), fps=8)
    println("wrote rigorous_denied_sweep.gif")
end
println("Vpass max=", round(vhi;digits=1), "  denied max=", round(dhi;digits=1))
rmprocs(workers())
