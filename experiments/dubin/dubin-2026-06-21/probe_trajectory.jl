using Pkg
Pkg.activate("experiments")
using ExperimentTools, Flux, JLD2, MarkovGames, MatrixAlphaZero
using POSGModels.Dubin, POSGModels.StaticArrays, Printf
const AZ = MatrixAlphaZero
const DIR = @__DIR__

s0() = JointDubinState(SA[1,1,deg2rad(45)], SA[8,7,deg2rad(180)])
Hbits(p) = (p=collect(Float64.(p)); s=sum(p); s>0 && (p./=s); -sum(x->x>0 ? x*log2(x) : 0.0, p))
nz(v) = round.(AZ.normalized_or_uniform(Float64.(v)); digits=3)
rr(v) = round.(Float64.(v); digits=2)

function main()
    game = DubinMG(V=(1.0,1.0))
    oracle = AZ.load_oracle(joinpath(DIR,"oracle_smoos.jld2"))
    s = s0()
    iters = [0,1,2,3,5,8,12,20,35,60,100,200,400,700,1000,1221]
    println("state = reference initial state\n")
    @printf("%-6s | %-22s H1  | %-22s | %-22s H2  | %-22s\n",
            "iter","π1 (strategy)","regret1","π2 (strategy)","regret2")
    println("-"^110)
    for it in iters
        f = joinpath(DIR,"models_smoos",@sprintf("oracle%04d.jld2",it))
        isfile(f) || (f = joinpath(DIR,"models_smoos","oracle$(it).jld2"))
        isfile(f) || continue
        Flux.loadmodel!(oracle, f)
        s1,s2 = AZ.state_strategy(oracle, game, s)
        r1,r2 = AZ.state_regret(oracle, game, s)
        @printf("%-6d | %-22s %.2f | %-22s | %-22s %.2f | %-22s\n",
                it, string(nz(s1)), Hbits(s1), string(rr(r1)),
                    string(nz(s2)), Hbits(s2), string(rr(r2)))
    end
end
main()
