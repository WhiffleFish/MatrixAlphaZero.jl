using Pkg
Pkg.activate("experiments")
using ExperimentTools, Flux, JLD2, MarkovGames, MatrixAlphaZero
using POSGModels.Dubin, POSGModels.StaticArrays, Random, Printf, Statistics
const AZ = MatrixAlphaZero
const DIR = @__DIR__

function load_oracle(iter)
    o = AZ.load_oracle(joinpath(DIR, "oracle_smoos.jld2"))
    Flux.loadmodel!(o, joinpath(DIR, "models_smoos", "oracle$(iter).jld2"))
    o
end
transfer_tau(iter; M, w) = (t=0.0; for _ in 1:iter; t = AZ.advance_transfer_tau(t, M, w); end; t)
s0() = JointDubinState(SA[1,1,deg2rad(45)], SA[8,7,deg2rad(180)])

function run_seeds(oracle, game, s; tw, tau, ϵ, K, iters=1000, max_depth=5)
    search = AZ.SMOOSSearch(; oracle, oos_iterations=iters, max_depth, transfer_weight=tw, τ=tau, ϵ=_->ϵ)
    x_argmax = Int[]; y_argmax = Int[]
    rx_argmax = Int[]; ry_argmax = Int[]
    xs = Vector{Float64}[]
    for k in 1:K
        Random.seed!(1000+k)
        yr, ys = AZ.fitted_smoos(search, game, s; ϵ)
        x = AZ.normalized_or_uniform(Float64.(ys[1])); y = AZ.normalized_or_uniform(Float64.(ys[2]))
        push!(x_argmax, argmax(x)); push!(y_argmax, argmax(y))
        push!(rx_argmax, argmax(Float64.(yr[1]))); push!(ry_argmax, argmax(Float64.(yr[2])))
        push!(xs, x)
    end
    return (; x_argmax, y_argmax, rx_argmax, ry_argmax, xmean=mean(xs), xstd=std(reduce(hcat,xs); dims=2)[:])
end

hist(a, n) = [count(==(i), a) for i in 1:n]

function main()
    game = DubinMG(V=(1.0,1.0)); oracle = load_oracle(1221)
    A1,A2 = actions(game); n1,n2 = length(A1), length(A2)
    K = 30
    states = [(s0(), "reference")]
    # add a downstream state
    rng = MersenneTwister(42); s = s0()
    for _ in 1:3; s,_ = @gen(:sp,:r)(game, s, (rand(rng,A1),rand(rng,A2)), rng); end
    push!(states, (s, "rollout-state-4"))

    for (st, name) in states
        s1,s2 = AZ.state_strategy(oracle, game, st)
        println("\n", "#"^76, "\nSTATE: $name")
        @printf("  network prior π1=%s  π2=%s\n",
                string(round.(AZ.normalized_or_uniform(Float64.(s1)); digits=3)),
                string(round.(AZ.normalized_or_uniform(Float64.(s2)); digits=3)))
        for ϵ in (0.1, 0.3)
            for (tw, lbl) in ((0.0,"tw=0 (no prior)"), (0.1,"tw=0.1 (train-like)"))
                tau = transfer_tau(1221; M=1000, w=tw)
                r = run_seeds(oracle, game, st; tw, tau, ϵ, K)
                @printf("  ε=%.1f %-20s | p1 strat argmax hist %s | p1 regret argmax hist %s\n",
                        ϵ, lbl, string(hist(r.x_argmax,n1)), string(hist(r.rx_argmax,n1)))
                @printf("  %26s | p1 mean strat %s  (per-action std %s)\n", "",
                        string(round.(r.xmean; digits=3)), string(round.(r.xstd; digits=3)))
                @printf("  %26s | p2 strat argmax hist %s (contrast)\n", "", string(hist(r.y_argmax,n2)))
            end
        end
    end
end
main()
