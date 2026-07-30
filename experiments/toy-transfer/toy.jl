# Toy testbed for regret/strategy transfer in regret-matching search.
#
# A tiny tabular simultaneous-move zero-sum Markov game ("trap game") is driven
# through the *actual* MCTSSearch/RegretMatchingSearch implementation with a
# tabular oracle, so exploitability of the search-induced Markov profile is
# exact (backward-induction best response) rather than a PPO approximation.
#
# Two oracle corruption models are supported:
#
#   :trap        — localized, confidently wrong prior at off-distribution states
#                  (one-hot strategy prior, hallucinated regret, and value
#                  estimates that confirm the wrong belief).
#   :calibrated  — global regression noise matched to the *measured* SDA
#                  regret-fit error profile, where the fitted average regret has
#                  a signal-to-noise ratio near one and its induced
#                  regret-matching direction is barely closer to the truth than
#                  a uniform prior. See `calibrate.jl`.
#
# Regret convention matches deployment: `state_regret` returns the average
# regret R̄ = Rᶜ_{T₁}/T₁, and the search injects `prior_scale · R̄`.
module ToyTransfer

using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Random
using Distributions
using Statistics
using LinearAlgebra

const AZ = MatrixAlphaZero

export TabularMG, trap_game, TrapSpec, NoiseSpec, ToyOracle, build_oracle,
       solve_game, exploitability, induced_profile, profile_summary,
       oracle_profile, ne_profile, decomposed_gaps, head_to_head,
       MAIN_ID, TRAP_ID, U_ID

# ---------------------------------------------------------------------------
# Generic tabular simultaneous-move zero-sum Markov game.
# States are integers 1:n; 0 is the terminal sink. All states share the same
# action counts (required by the search's global `actions(game)`).
# ---------------------------------------------------------------------------
struct TabularMG <: MG{Int, Tuple{Int, Int}}
    r  :: Vector{Matrix{Float64}}   # r[s][i, j] payoff to player 1
    sp :: Vector{Matrix{Int}}       # sp[s][i, j] successor id (0 = terminal)
    γ  :: Float64
end

nonterminal_states(g::TabularMG) = 1:length(g.r)

POMDPs.initialstate(::TabularMG) = Deterministic(1)
POMDPs.discount(g::TabularMG) = g.γ
POMDPs.isterminal(::TabularMG, s::Int) = s <= 0
POMDPs.actions(g::TabularMG) = (axes(g.r[1], 1), axes(g.r[1], 2))
POMDPs.convert_s(::Type{Vector{Float32}}, s::Int, ::TabularMG) = Float32[s]

function POMDPs.gen(g::TabularMG, s::Int, a::Tuple{Int, Int}, rng::Random.AbstractRNG=Random.default_rng())
    isterminal(g, s) && return (sp=0, r=0.0)
    return (sp=g.sp[s][a...], r=g.r[s][a...])
end

# ---------------------------------------------------------------------------
# Exact matrix-game solving: regret matching+ with linear averaging.
# ---------------------------------------------------------------------------
function solve_matrix(A::AbstractMatrix; iters::Int=50_000)
    n1, n2 = size(A)
    R1 = zeros(n1); R2 = zeros(n2)
    X = zeros(n1); Y = zeros(n2)
    wsum = 0.0
    x = fill(inv(n1), n1); y = fill(inv(n2), n2)
    for t in 1:iters
        u = x' * A * y
        r1 = A * y .- u
        r2 = u .- vec(x' * A)
        R1 .= max.(R1 .+ r1, 0.0)          # RM+
        R2 .= max.(R2 .+ r2, 0.0)
        X .+= t .* x; Y .+= t .* y; wsum += t   # linear averaging
        x = AZ.regret_matching_policy(R1)
        y = AZ.regret_matching_policy(R2)
    end
    x̄ = X ./ wsum; ȳ = Y ./ wsum
    return x̄, ȳ, x̄' * A * ȳ
end

# ---------------------------------------------------------------------------
# Trap game.
#
# ids: 1 root | 2..7 main subgames M[i,j] (i∈1:3, j∈1:2) | 8 trap gate T |
#      9..17 trap leaves U[i,j] | 0 terminal.
#
# Root: joint action (i, j): j ∈ {1,2} → M[i,j] with zero immediate reward;
#       j = 3 → T with immediate reward δ > 0 to P1 (P2 pays to enter the trap).
# M[i,j]: one-shot weighted-RPS matrix (mixed NE, value m[i,j]), then terminal.
# T: zero reward, sp[i,j] = U[i,j] — the effective stage matrix is γ·W where
#    W[i,j] = value(U[i,j]).
# U[i,j]: one-shot RPS + W[i,j], then terminal.
#
# At equilibrium P2 never enters the trap (value of the main columns < δ), so
# T and the U's are off-distribution; a best responder can always steer there.
# ---------------------------------------------------------------------------
rps(a, b, c) = [0.0 -a b; a 0.0 -c; -b c 0.0]   # NE ∝ (c, b, a), value 0

const MAIN_ID = [2 5; 3 6; 4 7]
const TRAP_ID = 8
const U_ID = [9 12 15; 10 13 16; 11 14 17]

Base.@kwdef struct TrapSpec
    δ  :: Float64 = 0.05                        # P2's cost to enter the trap
    W  :: Matrix{Float64} = rps(1.0, 2.0, 1.0)  # true trap stage matrix
    m  :: Matrix{Float64} = [0.15 -0.35; -0.30 0.10; -0.05 0.05]  # main values
    k1 :: Int = 3                               # P1's confidently-wrong action at T
    k2 :: Int = 1                               # P2's confidently-wrong action at T
    B_v       :: Float64 = 2.5    # value hallucination: V̂(U[k1,j]) = W[k1,j] + B_v
    sharpness :: Float64 = 0.95   # prior strategy mass on the wrong action
    ζ         :: Float64 = 2.0    # hallucinated positive regret scale at T
    γ         :: Float64 = 1.0
end

function trap_game(spec::TrapSpec=TrapSpec())
    n = 17
    r = [zeros(3, 3) for _ in 1:n]
    sp = [zeros(Int, 3, 3) for _ in 1:n]

    for i in 1:3, j in 1:3
        if j <= 2
            sp[1][i, j] = MAIN_ID[i, j]
        else
            sp[1][i, j] = TRAP_ID
            r[1][i, j] = spec.δ
        end
    end
    weights = [(1.0, 1.5, 0.5), (0.5, 1.0, 2.0), (2.0, 0.5, 1.0),
               (1.5, 0.5, 1.0), (1.0, 2.0, 1.5), (0.5, 2.0, 1.0)]
    for i in 1:3, j in 1:2
        a, b, c = weights[(j - 1) * 3 + i]
        r[MAIN_ID[i, j]] .= rps(a, b, c) .+ spec.m[i, j]
    end
    for i in 1:3, j in 1:3
        sp[TRAP_ID][i, j] = U_ID[i, j]
    end
    for i in 1:3, j in 1:3
        r[U_ID[i, j]] .= rps(1.0, 1.0, 1.0) .+ spec.W[i, j]
    end
    return TabularMG(r, sp, spec.γ)
end

# ---------------------------------------------------------------------------
# Exact solve, best response, and exploitability.
# ---------------------------------------------------------------------------
local_matrix(g::TabularMG, v::AbstractDict{Int, Float64}, s::Int) =
    [g.r[s][i, j] + g.γ * v[g.sp[s][i, j]] for i in axes(g.r[s], 1), j in axes(g.r[s], 2)]

function solve_game(g::TabularMG; iters::Int=50_000)
    v = Dict{Int, Float64}(0 => 0.0)
    profile = Dict{Int, NTuple{2, Vector{Float64}}}()
    q = Dict{Int, Matrix{Float64}}()
    for s in sort(collect(nonterminal_states(g)); rev=true)
        A = local_matrix(g, v, s)
        x, y, val = solve_matrix(A; iters)
        v[s] = val
        profile[s] = (x, y)
        q[s] = A
    end
    return (; v, profile, q)
end

function best_response_value(g::TabularMG, profile, player::Int)
    v = Dict{Int, Float64}(0 => 0.0)
    for s in sort(collect(nonterminal_states(g)); rev=true)
        A = local_matrix(g, v, s)
        x, y = profile[s]
        v[s] = player == 1 ? maximum(A * y) : minimum(vec(x' * A))
    end
    return v[1]
end

function profile_value(g::TabularMG, profile)
    v = Dict{Int, Float64}(0 => 0.0)
    for s in sort(collect(nonterminal_states(g)); rev=true)
        A = local_matrix(g, v, s)
        x, y = profile[s]
        v[s] = x' * A * y
    end
    return v[1]
end

# Saddle-point gap (NashConv) of a Markov profile: 0 at equilibrium.
function exploitability(g::TabularMG, profile)
    br1 = best_response_value(g, profile, 1)
    br2 = best_response_value(g, profile, 2)
    return (; gap = br1 - br2, br1, br2, self = profile_value(g, profile))
end

# ---------------------------------------------------------------------------
# Tabular oracle implementing the MatrixAlphaZero oracle interface.
# ---------------------------------------------------------------------------
struct ToyOracle
    v        :: Dict{Int, Float64}
    regret   :: Dict{Int, NTuple{2, Vector{Float64}}}
    strategy :: Dict{Int, NTuple{2, Vector{Float64}}}
end

sid(x::AbstractVector) = Int(round(Float64(only(x))))
const UNIF3 = fill(inv(3), 3)
const ZERO3 = zeros(3)

AZ.value(o::ToyOracle, x::AbstractVector) = Float32[get(o.v, sid(x), 0.0)]
AZ.value(o::ToyOracle, x::AbstractMatrix) =
    reshape(Float32[get(o.v, sid(col), 0.0) for col in eachcol(x)], 1, :)

AZ.batch_state_value(o::ToyOracle, ::MG, sv) = Float32[get(o.v, s, 0.0) for s in sv]
AZ.state_regret(o::ToyOracle, ::MG, s) = get(o.regret, s, (ZERO3, ZERO3))
AZ.state_strategy(o::ToyOracle, ::MG, s) = get(o.strategy, s, (UNIF3, UNIF3))
AZ.batch_state_regret(o::ToyOracle, game::MG, sv) = begin
    r = map(s -> AZ.state_regret(o, game, s), sv)
    (map(first, r), map(last, r))
end
AZ.batch_state_strategy(o::ToyOracle, game::MG, sv) = begin
    σ = map(s -> AZ.state_strategy(o, game, s), sv)
    (map(first, σ), map(last, σ))
end

# Value-only oracle: same critic, uniform strategy prior, zero regret prior.
struct ValueOnlyToyOracle
    inner :: ToyOracle
end

AZ.value(o::ValueOnlyToyOracle, x) = AZ.value(o.inner, x)
AZ.batch_state_value(o::ValueOnlyToyOracle, g::MG, sv) = AZ.batch_state_value(o.inner, g, sv)
AZ.state_regret(::ValueOnlyToyOracle, ::MG, s) = (ZERO3, ZERO3)
AZ.state_strategy(::ValueOnlyToyOracle, ::MG, s) = (UNIF3, UNIF3)
AZ.batch_state_regret(o::ValueOnlyToyOracle, g::MG, sv) =
    (fill(ZERO3, length(sv)), fill(ZERO3, length(sv)))
AZ.batch_state_strategy(o::ValueOnlyToyOracle, g::MG, sv) =
    (fill(UNIF3, length(sv)), fill(UNIF3, length(sv)))

# Zero oracle: SM-MCTS-A with no learned quantities at all.
struct ZeroToyOracle end

AZ.value(::ZeroToyOracle, x::AbstractVector) = Float32[0]
AZ.value(::ZeroToyOracle, x::AbstractMatrix) = zeros(Float32, 1, size(x, 2))
AZ.batch_state_value(::ZeroToyOracle, ::MG, sv) = zeros(Float32, length(sv))
AZ.state_regret(::ZeroToyOracle, ::MG, s) = (ZERO3, ZERO3)
AZ.state_strategy(::ZeroToyOracle, ::MG, s) = (UNIF3, UNIF3)
AZ.batch_state_regret(::ZeroToyOracle, ::MG, sv) =
    (fill(ZERO3, length(sv)), fill(ZERO3, length(sv)))
AZ.batch_state_strategy(::ZeroToyOracle, ::MG, sv) =
    (fill(UNIF3, length(sv)), fill(UNIF3, length(sv)))

# ---------------------------------------------------------------------------
# Oracle construction.
# ---------------------------------------------------------------------------

# Average regret and average strategy a converged RM solve would leave behind.
# The regret head's target is the *average* regret Rᶜ/T₁, matching the fitted
# target the deployment prior multiplies by `prior_scale`.
function fitted_rm_artifacts(A::AbstractMatrix; T1::Int=500)
    n1, n2 = size(A)
    R1 = zeros(n1); R2 = zeros(n2)
    X = zeros(n1); Y = zeros(n2)
    x = fill(inv(n1), n1); y = fill(inv(n2), n2)
    for _ in 1:T1
        u = x' * A * y
        R1 .= max.(R1 .+ (A * y .- u), 0.0)     # RM+, as the search uses
        R2 .= max.(R2 .+ (u .- vec(x' * A)), 0.0)
        X .+= x; Y .+= y
        x = AZ.regret_matching_policy(R1)
        y = AZ.regret_matching_policy(R2)
    end
    return (R1 ./ T1, R2 ./ T1), (X ./ T1, Y ./ T1)
end

sharp(k::Int, sharpness::Float64) =
    [i == k ? sharpness + (1 - sharpness) / 3 : (1 - sharpness) / 3 for i in 1:3]

# Global regression-noise model for the fitted heads. `σ_r` perturbs the regret
# direction, `α_r` pulls it toward uniform (the shrinkage a squared-error fit of
# a sparse nonnegative target produces), `σ_s` perturbs the strategy head, and
# `σ_v` is additive critic noise. `calibrate.jl` fits σ_r/α_r to the measured
# SDA test-split statistics.
Base.@kwdef struct NoiseSpec
    σ_r :: Float64 = 0.0
    α_r :: Float64 = 0.0
    σ_s :: Float64 = 0.0
    σ_v :: Float64 = 0.0
end

function noisy_regret(R̄::AbstractVector, spec::NoiseSpec, rng)
    base = AZ.regret_matching_policy(max.(R̄, 0.0))
    mass = sum(max.(R̄, 0.0))
    p = (1 - spec.α_r) .* base .+ spec.α_r ./ length(base) .+
        spec.σ_r .* abs.(randn(rng, length(base)))
    return (mass > 0 ? mass : 1.0) .* AZ.normalize_or_uniform!(p)
end

function noisy_strategy(σ̄::AbstractVector, spec::NoiseSpec, rng)
    p = σ̄ .+ spec.σ_s .* abs.(randn(rng, length(σ̄)))
    return AZ.normalize_or_uniform!(p)
end

"""
    build_oracle(g, spec, sol; corruption, noise, T1, seed)

Tabular oracle for the trap game.

`corruption` selects `:none`, `:trap` (localized confidently-wrong prior in the
off-distribution trap branch), `:calibrated` (global regression noise from
`noise`), or `:both`.
"""
function build_oracle(
        g::TabularMG,
        spec::TrapSpec,
        sol;
        corruption::Symbol=:trap,
        noise::NoiseSpec=NoiseSpec(),
        T1::Int=500,
        seed::Int=1,
    )
    corruption ∈ (:none, :trap, :calibrated, :both) ||
        throw(ArgumentError("Unknown corruption=$(corruption)"))
    rng = Random.MersenneTwister(seed)
    trap_corrupt = corruption ∈ (:trap, :both)
    noisy = corruption ∈ (:calibrated, :both)

    v̂ = Dict{Int, Float64}(0 => 0.0)
    regret = Dict{Int, NTuple{2, Vector{Float64}}}()
    strategy = Dict{Int, NTuple{2, Vector{Float64}}}()

    # Exact values everywhere first; corruption overrides below.
    for s in nonterminal_states(g)
        v̂[s] = sol.v[s]
    end
    if trap_corrupt
        Ŵ = copy(spec.W)
        Ŵ[spec.k1, :] .+= spec.B_v
        for i in 1:3, j in 1:3
            v̂[U_ID[i, j]] = Ŵ[i, j]
        end
        x̂T = sharp(spec.k1, spec.sharpness)
        ŷT = sharp(spec.k2, spec.sharpness)
        strategy[TRAP_ID] = (x̂T, ŷT)
        regret[TRAP_ID] = (spec.ζ .* x̂T, spec.ζ .* ŷT)
        v̂[TRAP_ID] = g.γ * (x̂T' * Ŵ * ŷT)   # belief-consistent trap value
    end

    # Fitted artifacts of the oracle's own believed local matrix at every state
    # whose prior was not already overridden by the trap corruption.
    for s in sort(collect(nonterminal_states(g)); rev=true)
        haskey(regret, s) && continue
        A = [g.r[s][i, j] + g.γ * v̂[g.sp[s][i, j]] for i in 1:3, j in 1:3]
        r̂, σ̂ = fitted_rm_artifacts(A; T1)
        regret[s] = r̂
        strategy[s] = σ̂
    end

    if noisy
        for s in nonterminal_states(g)
            r̂ = regret[s]
            σ̂ = strategy[s]
            regret[s] = (noisy_regret(r̂[1], noise, rng), noisy_regret(r̂[2], noise, rng))
            strategy[s] = (noisy_strategy(σ̂[1], noise, rng), noisy_strategy(σ̂[2], noise, rng))
            v̂[s] += noise.σ_v * randn(rng)
        end
    end
    return ToyOracle(v̂, regret, strategy)
end

# ---------------------------------------------------------------------------
# Induced Markov profile of a search configuration.
# ---------------------------------------------------------------------------
function induced_profile(params::AZ.MCTSSearch, g::TabularMG; nreps::Int=16, ϵ::Float64=0.1, rng_seed::Int=0)
    Random.seed!(rng_seed)
    profile = Dict{Int, NTuple{2, Vector{Float64}}}()
    for s in nonterminal_states(g)
        xs = zeros(3); ys = zeros(3)
        for _ in 1:nreps
            (x, y, _v), _info = AZ.search_info(params, g, s; ϵ)
            xs .+= x; ys .+= y
        end
        profile[s] = (xs ./ nreps, ys ./ nreps)
    end
    return profile
end

oracle_profile(o::ToyOracle, g::TabularMG) =
    Dict(s => (collect(Float64.(AZ.state_strategy(o, g, s)[1])),
               collect(Float64.(AZ.state_strategy(o, g, s)[2])))
         for s in nonterminal_states(g))

ne_profile(sol) = sol.profile

function profile_summary(g::TabularMG, profile; label="")
    e = exploitability(g, profile)
    return (; label, e.gap, e.br1, e.br2, e.self)
end

# Decompose the gap: replace one region's strategies by the exact NE to isolate
# where the exploitability comes from.
function decomposed_gaps(g::TabularMG, profile, sol)
    trap_states = (TRAP_ID, vec(U_ID)...)
    main_states = (1, vec(MAIN_ID)...)
    p_main = Dict(s => (s in trap_states ? sol.profile[s] : profile[s]) for s in nonterminal_states(g))
    p_trap = Dict(s => (s in main_states ? sol.profile[s] : profile[s]) for s in nonterminal_states(g))
    return (; gap_main = exploitability(g, p_main).gap,
              gap_trap = exploitability(g, p_trap).gap)
end

# ---------------------------------------------------------------------------
# Head-to-head cross-play between two search configurations.
#
# Returns the seat-balanced utility of `A` against `B`: the mean of A's payoff
# as player 1 and A's payoff as player 2 (negated), so a positive number favors
# A independently of which seat carries the game's asymmetry.
# ---------------------------------------------------------------------------
function play_episode(g::TabularMG, p1::AZ.MCTSSearch, p2::AZ.MCTSSearch, s::Int; ϵ, max_steps=8)
    total = 0.0
    disc = 1.0
    for _ in 1:max_steps
        isterminal(g, s) && break
        x, _, _ = AZ.search(p1, g, s; ϵ)
        _, y, _ = AZ.search(p2, g, s; ϵ)
        a = AZ.action_idx_from_probs(x, y)
        i, j = Tuple(a)
        total += disc * g.r[s][i, j]
        disc *= g.γ
        s = g.sp[s][i, j]
    end
    return total
end

function head_to_head(
        g::TabularMG,
        A::AZ.MCTSSearch,
        B::AZ.MCTSSearch;
        episodes::Int=200,
        ϵ::Float64=0.1,
        rng_seed::Int=0,
    )
    root = rand(Random.MersenneTwister(rng_seed), initialstate(g))
    # Common random numbers across the two seat orders: a solver played against
    # itself then scores exactly zero, so the reported advantage is not polluted
    # by the difference between two independent sampling streams.
    Random.seed!(rng_seed)
    a_as_p1 = [play_episode(g, A, B, root; ϵ) for _ in 1:episodes]
    Random.seed!(rng_seed)
    a_as_p2 = [play_episode(g, B, A, root; ϵ) for _ in 1:episodes]
    d = (a_as_p1 .- a_as_p2) ./ 2
    return (;
        balanced=mean(d),
        se=std(d) / sqrt(episodes),
        a_as_p1=mean(a_as_p1),
        a_as_p2=mean(a_as_p2),
    )
end

end # module
