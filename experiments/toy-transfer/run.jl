# Compare regret-transfer mechanisms on the toy trap game under *both* metrics
# that matter: exact exploitability (the thing the PPO best-response diagnostic
# approximates) and head-to-head cross-play against the value-only solver.
#
#   julia --project=. experiments/toy-transfer/run.jl
#   julia --project=. experiments/toy-transfer/run.jl --queries 50 --regimes calibrated
#
# Solvers, all running through the production MCTSSearch/RegretMatchingSearch:
#   zero        — SM-MCTS-A: V̂ = 0, uniform fallback, no transfer
#   value       — learned V̂ only, no transfer
#   warmstart   — current transfer: prior added into the node accumulators
#   capped      — read-time mixing, effective mass min(m_R, ρ·n_s)
#   gated       — capped × evidence gate on the prior pair's saddle gap
include(joinpath(@__DIR__, "toy.jl"))

using .ToyTransfer
using MatrixAlphaZero
using Printf
using Statistics

const AZ = MatrixAlphaZero
const TT = ToyTransfer

# Noise level fitted to the measured SDA regret-head error by `calibrate.jl`
# (TV and entropy of the induced regret-matching direction). `HALF_NOISE` is a
# deliberately more optimistic oracle so conclusions are not an artifact of one
# noise level.
const SDA_NOISE = TT.NoiseSpec(σ_r=2.3, α_r=0.10, σ_s=0.15, σ_v=0.05)
const HALF_NOISE = TT.NoiseSpec(σ_r=1.0, α_r=0.05, σ_s=0.07, σ_v=0.02)

const CAP_RATIO = 0.5
const GATE_TOL = 1.0

# Deployment uses RM+; `Vanilla` is available because plain RM has no clipping
# and therefore keeps a wrong warm start alive far longer, which is worth
# separating from the transfer mechanism itself.
rm_style(method::Symbol, update::Symbol) = AZ.RegretMatchingSearch(;
    backup=:mean, method = method === :plus ? AZ.Plus() : AZ.Vanilla(), update,
)

function make_search(
        kind::Symbol, oracle, queries::Int, max_depth::Int, prior_scale::Float64;
        method::Symbol=:plus, update::Symbol=:sampled,
    )
    common = (;
        tree_queries=queries,
        max_depth,
        search_style=rm_style(method, update),
        value_target=:search,
    )
    kind === :zero && return AZ.MCTSSearch(; oracle=TT.ZeroToyOracle(), common...)
    kind === :value &&
        return AZ.MCTSSearch(; oracle=TT.ValueOnlyToyOracle(oracle), common...)
    transfer = (;
        oracle,
        prior_scale,
        regret_prior_weight=1.0,
        statistic_prior_weight=0.0,
        prior_reach_power=1.0,
        common...,
    )
    # Current deployment: regret prior added into the accumulators, no strategy
    # mass, so the emitted average is Σₜσₜ/T₂ rather than the weighted average
    # the transfer lemma bounds.
    kind === :warmstart && return AZ.MCTSSearch(;
        transfer..., strategy_prior_weight=0.0, transfer_mode=:warmstart,
    )
    # Warmstart plus the theory-consistent strategy credit, isolating the
    # averaging fix from the read-time mass fix.
    kind === :warmstart_lemma && return AZ.MCTSSearch(;
        transfer..., strategy_prior_weight=1.0, transfer_mode=:warmstart,
    )
    kind === :capped && return AZ.MCTSSearch(;
        transfer..., strategy_prior_weight=0.0,
        transfer_mode=:capped, transfer_cap_ratio=CAP_RATIO,
    )
    kind === :gated && return AZ.MCTSSearch(;
        transfer..., strategy_prior_weight=0.0,
        transfer_mode=:gated, transfer_cap_ratio=CAP_RATIO,
        transfer_gate_tol=GATE_TOL,
    )
    kind === :gated_lemma && return AZ.MCTSSearch(;
        transfer..., strategy_prior_weight=1.0,
        transfer_mode=:gated, transfer_cap_ratio=CAP_RATIO,
        transfer_gate_tol=GATE_TOL,
    )
    # Uniform-tempered warm start at several tempering weights.
    m = match(r"^temper([0-9p]+)$", string(kind))
    if !isnothing(m)
        temper = parse(Float64, replace(m.captures[1], "p" => "."))
        return AZ.MCTSSearch(;
            transfer..., strategy_prior_weight=0.0,
            transfer_mode=:tempered, transfer_temper=temper,
        )
    end
    error("Unknown solver kind $(kind)")
end

const SOLVERS = (
    :zero, :value, :warmstart, :warmstart_lemma, :capped, :gated, :gated_lemma,
    :temper0p03, :temper0p1, :temper0p3, :temper1p0, :temper3p0,
)

function regime_oracle(game, spec, sol, regime::Symbol, seed::Int)
    regime === :clean && return TT.build_oracle(game, spec, sol; corruption=:none, seed)
    regime === :calibrated &&
        return TT.build_oracle(game, spec, sol; corruption=:calibrated, noise=SDA_NOISE, seed)
    regime === :calibrated_half &&
        return TT.build_oracle(game, spec, sol; corruption=:calibrated, noise=HALF_NOISE, seed)
    regime === :trap && return TT.build_oracle(game, spec, sol; corruption=:trap, seed)
    regime === :both &&
        return TT.build_oracle(game, spec, sol; corruption=:both, noise=SDA_NOISE, seed)
    error("Unknown regime $(regime)")
end

function run_regime(
        game, spec, sol, regime::Symbol;
        queries::Int, max_depth::Int, prior_scale::Float64,
        nreps::Int, episodes::Int, seeds, method::Symbol, update::Symbol,
    )
    gaps = Dict(k => Float64[] for k in SOLVERS)
    gap_main = Dict(k => Float64[] for k in SOLVERS)
    gap_trap = Dict(k => Float64[] for k in SOLVERS)
    h2h = Dict(k => Float64[] for k in SOLVERS)
    raw_gaps = Float64[]

    for seed in seeds
        oracle = regime_oracle(game, spec, sol, regime, seed)
        push!(raw_gaps, TT.exploitability(game, TT.oracle_profile(oracle, game)).gap)
        searches = Dict(
            k => make_search(k, oracle, queries, max_depth, prior_scale; method, update)
            for k in SOLVERS
        )
        for k in SOLVERS
            prof = TT.induced_profile(
                searches[k], game; nreps, ϵ=0.1, rng_seed=Int(hash((regime, k, seed)) % 100_000),
            )
            push!(gaps[k], TT.exploitability(game, prof).gap)
            d = TT.decomposed_gaps(game, prof, sol)
            push!(gap_main[k], d.gap_main)
            push!(gap_trap[k], d.gap_trap)
            # head-to-head against the value-only solver (positive favors k)
            r = TT.head_to_head(
                game, searches[k], searches[:value];
                episodes, ϵ=0.1, rng_seed=Int(hash((regime, k, seed, :h2h)) % 100_000),
            )
            push!(h2h[k], r.balanced)
        end
    end

    n = length(seeds)
    paired(a, b) = (d = a .- b; (; delta=mean(d), se=n > 1 ? std(d) / sqrt(n) : 0.0))
    @printf("\n=== regime %s | %s | update=%s | queries=%d prior_scale=%.2f depth=%d | %d oracle seeds ===\n",
            regime, method === :plus ? "RM+" : "RM", update, queries, prior_scale, max_depth, n)
    @printf("raw prior profile gap = %.3f\n", mean(raw_gaps))
    @printf("%-16s %16s %20s %10s %10s %18s\n",
            "solver", "exact gap (↓)", "paired Δgap vs value", "gap_main", "gap_trap",
            "h2h vs value (↑)")
    for k in SOLVERS
        d = paired(gaps[k], gaps[:value])
        @printf("%-16s %8.3f ± %-5.3f %10.4f ± %-7.4f %10.3f %10.3f %10.4f ± %-6.4f\n",
                k, mean(gaps[k]), std(gaps[k]) / sqrt(n),
                d.delta, d.se,
                mean(gap_main[k]), mean(gap_trap[k]),
                mean(h2h[k]), std(h2h[k]) / sqrt(n))
    end
    println("(paired Δgap < 0 means less exploitable than the value-only solver on the")
    println(" same oracle draw; h2h > 0 means it also wins the cross-play.)")
    return (; gaps, h2h)
end

function main(args)
    queries = 100
    max_depth = 2
    nreps = 16
    episodes = 200
    nseeds = 5
    method = :plus
    update = :sampled
    regimes = [:clean, :calibrated_half, :calibrated, :trap, :both]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--queries"
            queries = parse(Int, args[i + 1]); i += 2
        elseif a == "--depth"
            max_depth = parse(Int, args[i + 1]); i += 2
        elseif a == "--nreps"
            nreps = parse(Int, args[i + 1]); i += 2
        elseif a == "--episodes"
            episodes = parse(Int, args[i + 1]); i += 2
        elseif a == "--seeds"
            nseeds = parse(Int, args[i + 1]); i += 2
        elseif a == "--update"
            update = Symbol(args[i + 1]); i += 2
        elseif a == "--method"
            method = Symbol(args[i + 1]); i += 2
        elseif a == "--regimes"
            regimes = Symbol.(split(args[i + 1], ",")); i += 2
        else
            error("unknown arg $(a)")
        end
    end
    # Deployment keeps the prior mass at 5% of the online query budget.
    prior_scale = 0.05 * queries

    spec = TT.TrapSpec()
    game = TT.trap_game(spec)
    sol = TT.solve_game(game)
    @assert sol.profile[1][2][3] < 1e-3 "P2 should avoid the trap at equilibrium"
    @printf("game value v* = %+.4f (δ = %.2f);  exact-NE gap = %.5f\n",
            sol.v[1], spec.δ, TT.exploitability(game, TT.ne_profile(sol)).gap)

    for regime in regimes
        run_regime(
            game, spec, sol, regime;
            queries, max_depth, prior_scale, nreps, episodes, seeds=1:nseeds, method, update,
        )
        flush(stdout)
    end
end

main(ARGS)
