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
        "--smooth";  action=:store_true;           help="apply light 3x3 display smoothing"
    end
    return parse_args(s)
end
const A = cli()

# light 3x3 smoothing (edge-replicated)
function smooth3(M)
    m, n = size(M); B = similar(M, Float64)
    for j in 1:n, i in 1:m
        acc = 0.0; c = 0
        for dj in -1:1, di in -1:1
            acc += M[clamp(i+di,1,m), clamp(j+dj,1,n)]; c += 1
        end
        B[i,j] = acc / c
    end
    return B
end
sm(M) = A["smooth"] ? smooth3(M) : M

d = load(A["infile"])
sep_ax, alt_ax, phis = d["sep_ax"], d["alt_ax"], d["phis"]
Veq, Vpass, denied = d["Veq"], d["Vpass"], d["denied"]
nframes = length(phis)

vhi = maximum(m -> maximum(m), vcat(Veq, Vpass))
dhi = maximum(m -> maximum(abs, m), denied)

function frame_plot(k)
    φdeg = round(Int, rad2deg(phis[k]))
    if A["panel"] == "denied"
        w = A["width"] == 0 ? 640 : A["width"]
        heatmap(sep_ax, alt_ax, sm(denied[k]), c=:thermal, clims=(0,dhi),
                xlabel="Δν (deg)", ylabel="Δa (km)", size=(w, round(Int, w*0.78)),
                title="opportunity denied by evasion  ($(φdeg)° from Sun)",
                left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    else
        w = A["width"] == 0 ? 1200 : A["width"]
        p1 = heatmap(sep_ax, alt_ax, Veq[k], c=:magma, clims=(0,vhi),
                     title="equilibrium V (net)", xlabel="Δν (deg)", ylabel="Δa (km)")
        p2 = heatmap(sep_ax, alt_ax, sm(Vpass[k]), c=:magma, clims=(0,vhi),
                     title="V vs passive target", xlabel="Δν (deg)")
        p3 = heatmap(sep_ax, alt_ax, sm(denied[k]), c=:thermal, clims=(0,dhi),
                     title="opportunity denied by evasion", xlabel="Δν (deg)")
        plot(p1, p2, p3, layout=(1,3), size=(w, round(Int, w*0.32)),
             bottom_margin=6Plots.mm, left_margin=6Plots.mm,
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
