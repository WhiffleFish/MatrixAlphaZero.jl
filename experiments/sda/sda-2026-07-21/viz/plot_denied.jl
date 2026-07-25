# Local renderer for the rigorous opportunity-denied sweep. Reads the raw arrays
# produced by rigorous_denied.jl (rigorous_denied_sweep.jld2) and builds the gif
# / static frame. Runs on a laptop — no Distributed, no compute, just plotting,
# so the headless-server GR segfault never enters the picture.
#
#   julia plot_denied.jl                 # 3-panel gif, defaults
#   julia plot_denied.jl --panel denied  # compact single-panel gif (smaller file)
#   julia plot_denied.jl --help
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..", "..", "..")); io=devnull)
using JLD2, Plots, ArgParse
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

const FIGDIR = joinpath(@__DIR__, "figs")

function cli()
    s = ArgParseSettings(description="Render the opportunity-denied sweep from saved arrays.")
    @add_arg_table! s begin
        "--infile";  arg_type=String; default=joinpath(FIGDIR, "rigorous_denied_sweep.jld2")
        "--panel";   arg_type=String; default="all"; help="which panels: all | denied"
        "--fps";     arg_type=Int;    default=8
        "--width";   arg_type=Int;    default=0;   help="frame width px (0 => auto by panel)"
        "--sigma";   arg_type=Float64; default=2.0
            help="Gaussian spatial smoothing width in grid cells (0 => none). " *
                 "The Monte-Carlo estimate has per-cell stderr ~6.7 while the true " *
                 "surface is smooth, so sigma~2 buys ~10x variance reduction free."
        "--downsample"; arg_type=Int; default=1
            help="average NxN cell blocks before plotting (2 => halve resolution, halve noise)"
        "--tsigma";  arg_type=Float64; default=0.0
            help="Gaussian smoothing across frames, in frames (0 => none). Use sparingly: " *
                 "occlusion boundaries genuinely move between frames."
    end
    return parse_args(s)
end
const A = cli()

# --- display smoothing -------------------------------------------------------
# Legitimate here because the underlying value surface is physically smooth in
# (Δν, Δa) -- the speckle is sampling noise, not structure. Raw arrays are kept
# in the .jld2; everything below is display-only.

function gaussian_blur(M, σ)
    σ <= 0 && return Float64.(M)
    r = max(1, ceil(Int, 3σ))
    k = [exp(-(d^2)/(2σ^2)) for d in -r:r]; k ./= sum(k)
    m, n = size(M)
    tmp = zeros(Float64, m, n); out = zeros(Float64, m, n)
    for j in 1:n, i in 1:m       # blur along rows
        acc = 0.0
        for (t, d) in enumerate(-r:r)
            acc += k[t] * M[clamp(i+d, 1, m), j]
        end
        tmp[i,j] = acc
    end
    for j in 1:n, i in 1:m       # blur along cols
        acc = 0.0
        for (t, d) in enumerate(-r:r)
            acc += k[t] * tmp[i, clamp(j+d, 1, n)]
        end
        out[i,j] = acc
    end
    return out
end

function block_average(M, b)
    b <= 1 && return Float64.(M)
    m, n = size(M)
    mo, no = fld(m, b), fld(n, b)
    B = zeros(Float64, mo, no)
    for j in 1:no, i in 1:mo
        B[i,j] = mean(@view M[(i-1)*b+1:i*b, (j-1)*b+1:j*b])
    end
    return B
end

# axis values after block averaging
function block_axis(ax, b)
    b <= 1 && return collect(ax)
    no = fld(length(ax), b)
    [mean(ax[(i-1)*b+1:i*b]) for i in 1:no]
end

sm(M) = gaussian_blur(block_average(M, A["downsample"]), A["sigma"])

using Statistics
d = load(A["infile"])
sep_ax_raw, alt_ax_raw, phis = d["sep_ax"], d["alt_ax"], d["phis"]
Veq_raw, Vpass_raw, denied_raw = d["Veq"], d["Vpass"], d["denied"]
nframes = length(phis)

# optional smoothing across frames (before spatial smoothing)
function temporal_blur(F, σ)
    σ <= 0 && return F
    r = max(1, ceil(Int, 3σ))
    k = [exp(-(dd^2)/(2σ^2)) for dd in -r:r]; k ./= sum(k)
    n = length(F)
    return [sum(k[t] * F[mod1(i + dd, n)] for (t, dd) in enumerate(-r:r)) for i in 1:n]
end

Vpass_t  = temporal_blur(Vpass_raw,  A["tsigma"])
denied_t = temporal_blur(denied_raw, A["tsigma"])

# apply display pipeline once, up front
sep_ax = block_axis(sep_ax_raw, A["downsample"])
alt_ax = block_axis(alt_ax_raw, A["downsample"])
Veq    = [sm(M) for M in Veq_raw]
Vpass  = [sm(M) for M in Vpass_t]
denied = [sm(M) for M in denied_t]

println("display pipeline: downsample=", A["downsample"], "  sigma=", A["sigma"],
        "  tsigma=", A["tsigma"], "  =>  grid ", size(Veq[1]))

vhi = maximum(m -> maximum(m), vcat(Veq, Vpass))
dhi = maximum(m -> maximum(abs, m), denied)

function frame_plot(k)
    φdeg = round(Int, rad2deg(phis[k]))
    if A["panel"] == "denied"
        w = A["width"] == 0 ? 640 : A["width"]
        heatmap(sep_ax, alt_ax, denied[k], c=:thermal, clims=(0,dhi),
                xlabel="Δν (deg)", ylabel="Δa (km)", size=(w, round(Int, w*0.78)),
                title="opportunity denied by evasion  ($(φdeg)° from Sun)",
                left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    else
        w = A["width"] == 0 ? 1200 : A["width"]
        p1 = heatmap(sep_ax, alt_ax, Veq[k], c=:magma, clims=(0,vhi),
                     title="equilibrium V (net)", xlabel="Δν (deg)", ylabel="Δa (km)")
        p2 = heatmap(sep_ax, alt_ax, Vpass[k], c=:magma, clims=(0,vhi),
                     title="V vs passive target", xlabel="Δν (deg)")
        p3 = heatmap(sep_ax, alt_ax, denied[k], c=:thermal, clims=(0,dhi),
                     title="opportunity denied by evasion", xlabel="Δν (deg)")
        plot(p1, p2, p3, layout=(1,3), size=(w, round(Int, w*0.36)),
             bottom_margin=6Plots.mm, left_margin=6Plots.mm, top_margin=7Plots.mm,
             plot_title="orbital position $(φdeg)° from Sun")
    end
end

tag = A["panel"] == "denied" ? "denied" : "sweep"
if nframes == 1
    frame_plot(1); savefig(joinpath(FIGDIR, "rigorous_denied_frame.png"))
    println("wrote rigorous_denied_frame.png")
else
    anim = @animate for k in 1:nframes
        frame_plot(k)
    end
    out = joinpath(FIGDIR, "rigorous_denied_$(tag).gif")
    gif(anim, out, fps=A["fps"])
    println("wrote ", out)
end
