using Pkg
Pkg.activate("experiments")

using JLD2, Flux, Statistics, Random
using MarkovGames, MatrixAlphaZero, POMDPTools
using POSGModels.Tron, POSGModels.StaticArrays
using ExperimentTools
const AZ = MatrixAlphaZero
const Tools = ExperimentTools
const TronTools = ExperimentTools.Tron

game = TronMG()
s0 = game.initialstate
n_s = 8 + 2 * game.width * game.height
width = 32
γ = MarkovGames.discount(game)

const MODELDIR = joinpath(@__DIR__, "models_regret_matching")

function init_oracle(n_s, width; value_weight=1f0)
    trunk = Chain(Dense(n_s => width, tanh), Dense(width => width, tanh))
    critic = Chain(Dense(width => width, tanh), Dense(width => 1))
    return AZ.CriticOnly(trunk, critic; value_weight)
end

function load_oracle(iter::Int)
    path = joinpath(MODELDIR, "oracle" * lpad(string(iter), 4, '0') * ".jld2")
    ms = JLD2.load(path, "model_state")
    o = init_oracle(n_s, width)
    Flux.loadmodel!(o, ms)
    return o
end

vscalar(o, s) = (v = AZ.state_value(o, game, s); v isa AbstractArray ? Float64(v[1]) : Float64(v))

function make_search(oracle; tree_queries, max_depth, eps=0.0)
    AZ.MCTSSearch(;
        oracle, tree_queries, max_depth, max_time = Inf,
        ϵ = _ -> eps,
        search_style = AZ.RegretMatchingSearch(; backup = :sample),
        value_target = :search,
    )
end

function eval_vs_heuristic(oracle; tree_queries, max_depth, eps=0.0, n=30, max_steps=120)
    search = make_search(oracle; tree_queries, max_depth, eps)
    planner = AZ.AlphaZeroPlanner(game, search)
    az_p1 = Tools.JointPolicy(Tools.SinglePlayerAlphaZeroPolicy(planner, 1), TronTools.tron_heuristic(game, 2))
    az_p2 = Tools.JointPolicy(TronTools.tron_heuristic(game, 1), Tools.SinglePlayerAlphaZeroPolicy(planner, 2))
    ev(jp) = Tools.evaluate_joint_policy(
        game, jp, n; max_steps, initialstates = fill(s0, n),
        show_progress = false, parallel = false,
        accumulators = (StepCount(), TronTools.TronOutcome()),
        batch_accumulators = (
            MeanResult(:steps; name = :mean_steps),
            RateResult(:p1_win), RateResult(:p2_win), RateResult(:draw),
        ),
    )
    r1 = ev(az_p1); r2 = ev(az_p2)
    return (
        reward_p1 = r1.reward[1], win_p1 = r1.p1_win_rate, steps_p1 = r1.mean_steps,
        reward_p2 = r2.reward[2], win_p2 = r2.p2_win_rate, steps_p2 = r2.mean_steps,
    )
end

# ---- collect (V̂, Voronoi-diff, eventual outcome) along trajectories of a joint policy ----
function voronoi_diff(s)
    occ = s.trail1 | s.trail2
    mine, theirs, _ = TronTools._voronoi_scores(
        game, occ, (Int(s.p1[1]), Int(s.p1[2])), (Int(s.p2[1]), Int(s.p2[2])))
    return mine - theirs   # >0 => p1 controls more territory
end

function collect_states(oracle, jp; n=25, max_steps=120, rng=MersenneTwister(1))
    V = Float64[]; Vor = Float64[]; Out = Float64[]  # Out = p1-perspective discounted return-to-go
    for _ in 1:n
        s = s0; traj = JointTronState[]
        outcome = 0.0; T = 0
        for t in 1:max_steps
            if isterminal(game, s); outcome = Float64(s.outcome); T = t; break; end
            push!(traj, s)
            a = rand(rng, MarkovGames.behavior(jp, s))
            s = rand(rng, transition(game, s, a))
            T = t
        end
        L = length(traj)
        for (k, st) in enumerate(traj)
            push!(V, vscalar(oracle, st))
            push!(Vor, voronoi_diff(st))
            push!(Out, γ^(L - k) * outcome)   # discounted return-to-go, p1 perspective
        end
    end
    return V, Vor, Out
end

pearson(x, y) = (length(x) < 3) ? NaN : cor(x, y)
signagree(x, y) = mean((sign.(x) .== sign.(y))[ (x .!= 0) .& (y .!= 0) ])

println("discount γ = $γ, board $(game.width)x$(game.height), n_s=$n_s")

# =========================================================================
# 1) FVI convergence probe: does depth matter? (does it matter LESS over training?)
# =========================================================================
println("\n### 1. Depth sensitivity across training (queries=500, eps=0, n=30) ###")
println("iter |  d=1 (r_p1/r_p2)   d=5 (r_p1/r_p2)   d=25 (r_p1/r_p2)")
for it in (100, 600, 1221)
    o = load_oracle(it)
    row = String[]
    for d in (1, 5, 25)
        r = eval_vs_heuristic(o; tree_queries=500, max_depth=d, eps=0.0, n=30)
        push!(row, "$(round(r.reward_p1;digits=2))/$(round(r.reward_p2;digits=2))")
    end
    println("$(lpad(it,4)) |  $(rpad(row[1],16))  $(rpad(row[2],16))  $(row[3])")
end

# =========================================================================
# 2) Search-budget sensitivity at fixed depth=5 (final ckpt): root converged?
# =========================================================================
println("\n### 2. Query sensitivity at depth=5, final ckpt (eps=0, n=30) ###")
let o = load_oracle(1221)
    for q in (100, 500, 2000, 8000)
        r = eval_vs_heuristic(o; tree_queries=q, max_depth=5, eps=0.0, n=30)
        println("  q=$(lpad(q,5)): reward_p1=$(round(r.reward_p1;digits=3)) win_p1=$(round(r.win_p1;digits=3))  reward_p2=$(round(r.reward_p2;digits=3)) win_p2=$(round(r.win_p2;digits=3))")
    end
    # eps sanity (match training-eval eps=0.1)
    r = eval_vs_heuristic(o; tree_queries=500, max_depth=5, eps=0.1, n=30)
    println("  [eps=0.1, q=500, d=5]: reward_p1=$(round(r.reward_p1;digits=3)) reward_p2=$(round(r.reward_p2;digits=3))  (compare wandb ~ -0.65)")
end

# =========================================================================
# 3) Value quality: does V̂ track outcome & territory? in-dist vs off-dist
# =========================================================================
println("\n### 3. Value function quality (final ckpt 1221) ###")
let o = load_oracle(1221)
    search = make_search(o; tree_queries=500, max_depth=5, eps=0.0)
    planner = AZ.AlphaZeroPlanner(game, search)
    az_self = Tools.JointPolicy(Tools.SinglePlayerAlphaZeroPolicy(planner, 1), Tools.SinglePlayerAlphaZeroPolicy(planner, 2))
    heur_both = TronTools.tron_heuristic_joint_policy(game)
    az_vs_h = Tools.JointPolicy(Tools.SinglePlayerAlphaZeroPolicy(planner, 1), TronTools.tron_heuristic(game, 2))

    for (name, jp, nn) in (("AZ self-play (train dist)", az_self, 20),
                           ("heuristic-vs-heuristic (off dist)", heur_both, 25),
                           ("AZ-p1 vs heuristic-p2 (eval dist)", az_vs_h, 25))
        V, Vor, Out = collect_states(o, jp; n=nn)
        println("  [$name]  states=$(length(V))")
        println("      V̂:        mean=$(round(mean(V);digits=3)) std=$(round(std(V);digits=3)) min=$(round(minimum(V);digits=3)) max=$(round(maximum(V);digits=3))")
        println("      corr(V̂, discounted outcome) = $(round(pearson(V,Out);digits=3))")
        println("      corr(V̂, Voronoi territory)  = $(round(pearson(V,Vor);digits=3))")
        println("      sign(V̂)==sign(outcome)      = $(round(signagree(V,Out);digits=3))")
        println("      sign(Voronoi)==sign(outcome)= $(round(signagree(Vor,Out);digits=3))  (heuristic-proxy ceiling)")
    end
end

println("\nDONE")
