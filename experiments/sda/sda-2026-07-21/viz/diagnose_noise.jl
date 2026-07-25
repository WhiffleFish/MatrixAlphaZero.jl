using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..", "..", "..")); io=devnull)
using JLD2, Statistics

d = load(joinpath(@__DIR__, "figs", "rigorous_denied_sweep.jld2"))
Veq, Vpass, denied = d["Veq"], d["Vpass"], d["denied"]
phis = d["phis"]; nf = length(phis)
println("frames=$(nf)  grid=$(size(Vpass[1]))  eps=$(d["eps"])")

# Noise proxy: high-frequency content = deviation from a 3x3 local mean.
# Signal proxy: spatial std of the smoothed field.
function smooth3(M)
    m, n = size(M); B = similar(M, Float64)
    for j in 1:n, i in 1:m
        acc = 0.0; c = 0
        for dj in -1:1, di in -1:1
            acc += M[clamp(i+di,1,m), clamp(j+dj,1,n)]; c += 1
        end
        B[i,j] = acc/c
    end
    B
end

for (name, F) in (("Vpass", Vpass), ("denied", denied), ("Veq", Veq))
    hf = Float64[]; sig = Float64[]
    for M in F
        S = smooth3(M)
        append!(hf, vec(M .- S))
        push!(sig, std(S))
    end
    println(rpad(name, 8), " spatial-signal std=", round(mean(sig); digits=2),
            "   high-freq noise std=", round(std(hf); digits=2),
            "   SNR=", round(mean(sig)/std(hf); digits=2))
end

# Frame-to-frame jitter: neighboring phis should be nearly identical physically,
# so differences between adjacent frames are almost pure sampling noise.
jit = Float64[]
for k in 1:nf-1
    append!(jit, vec(Vpass[k+1] .- Vpass[k]))
end
println("adjacent-frame Vpass diff std=", round(std(jit); digits=2))

# Implied per-cell standard error, and episodes needed for a target SE.
se = std(jit)/sqrt(2)
println("implied per-cell stderr ≈ ", round(se; digits=2), " (with eps=", d["eps"], ")")
println("=> per-episode std ≈ ", round(se*sqrt(d["eps"]); digits=1))
for target in (4.0, 2.0, 1.0)
    need = (se*sqrt(d["eps"])/target)^2
    println("   eps needed for SE=$(target): ", ceil(Int, need),
            "  (", round(ceil(need)/d["eps"]; digits=1), "x current cost)")
end
