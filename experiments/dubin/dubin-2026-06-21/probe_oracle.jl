using Pkg
Pkg.activate("experiments")

using ExperimentTools
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random
using Printf

const AZ = MatrixAlphaZero
const EXPERIMENT_DIR = @__DIR__

# ---- replicate the experiment's loading / config ----
function load_checkpoint_oracle(iter::Int)
    oracle_file = joinpath(EXPERIMENT_DIR, "oracle_smoos.jld2")
    checkpoint = joinpath(EXPERIMENT_DIR, "models_smoos", "oracle$(iter).jld2")
    oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(oracle, checkpoint)
    return oracle
end

transfer_tau(iter; oos_iterations, transfer_weight) = begin
    tau = 0.0
    for _ in 1:iter
        tau = AZ.advance_transfer_tau(tau, oos_iterations, transfer_weight)
    end
    tau
end

initial_dubin_state() = JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

entropy_bits(p) = begin
    p = collect(Float64.(p))
    s = sum(p); s > 0 && (p ./= s)
    -sum(x -> x > 0 ? x*log2(x) : 0.0, p)
end

fmt(v) = join([@sprintf("%.3f", x) for x in v], " ")

function describe_prior(oracle, game, s, label)
    s1, s2 = AZ.state_strategy(oracle, game, s)
    r1, r2 = AZ.state_regret(oracle, game, s)
    v = AZ.oracle_state_value(oracle, game, s)
    A1, A2 = actions(game)
    n1, n2 = length(A1), length(A2)
    println("── $label   value=$(@sprintf("%.3f", v))")
    println("   prior π1 (n=$n1): [$(fmt(AZ.normalized_or_uniform(Float64.(s1))))]  H=$(@sprintf("%.3f", entropy_bits(s1)))/$(@sprintf("%.3f", log2(n1))) bits")
    println("   prior π2 (n=$n2): [$(fmt(AZ.normalized_or_uniform(Float64.(s2))))]  H=$(@sprintf("%.3f", entropy_bits(s2)))/$(@sprintf("%.3f", log2(n2))) bits")
    println("   regret r1: [$(fmt(r1))]")
    println("   regret r2: [$(fmt(r2))]")
end

function smoos_acted_entropy(oracle, game, s; tw, tau, ϵ, iters=1000, max_depth=5, seed=1)
    search = AZ.SMOOSSearch(; oracle, oos_iterations=iters, max_depth, transfer_weight=tw, τ=tau, ϵ=_->ϵ)
    Random.seed!(seed)
    yr, ys = AZ.fitted_smoos(search, game, s; ϵ)
    x = AZ.normalized_or_uniform(Float64.(ys[1]))
    y = AZ.normalized_or_uniform(Float64.(ys[2]))
    return x, y
end

function main()
    game = DubinMG(V=(1.0, 1.0))
    oracle = load_checkpoint_oracle(1221)
    A1, A2 = actions(game)
    n1, n2 = length(A1), length(A2)
    println("Dubin actions: n1=$n1 n2=$n2 (max entropy p1=$(@sprintf("%.3f", log2(n1))) bits)")

    # states: reference + a few along a random rollout
    s0 = initial_dubin_state()
    rng = MersenneTwister(42)
    states = [s0]
    s = s0
    for _ in 1:3
        isterminal(game, s) && break
        a = (rand(rng, A1), rand(rng, A2))
        s, _ = @gen(:sp, :r)(game, s, a, rng)
        push!(states, s)
    end

    weights = [(0.0, "tw=0.0 (oos_value)"), (0.1, "tw=0.1 (oos_transfer, BR run)"), (1.0, "tw=1.0")]
    for (idx, st) in enumerate(states)
        println("\n", "="^78)
        println("STATE $idx", idx == 1 ? "  (reference initial state)" : "")
        describe_prior(oracle, game, st, "network prior")
        for ϵ in (0.3, 0.1)
            println("   --- SMOOS acted root strategy (ϵ=$ϵ, 1000 iters) ---")
            for (tw, lbl) in weights
                tau = transfer_tau(1221; oos_iterations=1000, transfer_weight=tw)
                x, y = smoos_acted_entropy(oracle, game, st; tw, tau, ϵ)
                @printf("   %-32s π1=[%s] H1=%.3f | π2 H2=%.3f  (τ=%.1f)\n",
                        lbl, fmt(x), entropy_bits(x), entropy_bits(y), tau)
            end
        end
    end
end

main()
