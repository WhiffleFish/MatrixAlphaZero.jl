# Shared harness for the transfer-mode comparison: SDA game, oracles, solver
# construction, an exact truncated best-response computation, and head-to-head
# cross-play.
#
# Deliberately avoids ExperimentTools so the script runs without the Wandb /
# PythonCall / Conda bootstrap.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributions
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Printf
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics
using Statistics

const AZ = MatrixAlphaZero
const EXPERIMENT_DIR = @__DIR__
const SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const BASELINE_MODELS =
    joinpath(EXPERIMENT_DIR, "regret_fit_results_softplus_long", "models.jld2")
const HURDLE_MODELS =
    joinpath(EXPERIMENT_DIR, "regret_fit_results_final_iter", "models.jld2")

include(joinpath(EXPERIMENT_DIR, "initial_state.jl"))

make_game() = SNRGameSimple(altitude_bounds=(100e3, 2e7))

# ---------------------------------------------------------------------------
# Oracle wrappers
# ---------------------------------------------------------------------------
struct ValueOnlySearchOracle{O}
    oracle::O
    na::NTuple{2,Int}
end

ValueOnlySearchOracle(game::MG, oracle) =
    ValueOnlySearchOracle(oracle, Tuple(length.(actions(game))))

uniform_pair(na::NTuple{2,Int}) =
    (fill(Float32(inv(na[1])), na[1]), fill(Float32(inv(na[2])), na[2]))

AZ.value(o::ValueOnlySearchOracle, x) = AZ.value(o.oracle, x)
AZ.state_value(o::ValueOnlySearchOracle, game, s) = AZ.state_value(o.oracle, game, s)
AZ.batch_state_value(o::ValueOnlySearchOracle, game, states) =
    AZ.batch_state_value(o.oracle, game, states)
AZ.state_policy(o::ValueOnlySearchOracle, game, s) = uniform_pair(o.na)
AZ.state_strategy(o::ValueOnlySearchOracle, game, s) = uniform_pair(o.na)
AZ.batch_state_policy(o::ValueOnlySearchOracle, game, states) = (
    fill(Float32(inv(o.na[1])), o.na[1], length(states)),
    fill(Float32(inv(o.na[2])), o.na[2], length(states)),
)
AZ.batch_state_strategy(o::ValueOnlySearchOracle, game, states) =
    AZ.batch_state_policy(o, game, states)
AZ.state_regret(o::ValueOnlySearchOracle, game, s) =
    (zeros(Float32, o.na[1]), zeros(Float32, o.na[2]))
AZ.batch_state_regret(o::ValueOnlySearchOracle, game, states) = (
    zeros(Float32, o.na[1], length(states)),
    zeros(Float32, o.na[2], length(states)),
)

struct ZeroSearchOracle
    na::NTuple{2,Int}
end

ZeroSearchOracle(game::MG) = ZeroSearchOracle(Tuple(length.(actions(game))))

AZ.value(::ZeroSearchOracle, x::AbstractVector) = Float32[0]
AZ.value(::ZeroSearchOracle, x::AbstractMatrix) = zeros(Float32, 1, size(x, 2))
AZ.state_value(::ZeroSearchOracle, game, s) = 0.0f0
AZ.batch_state_value(::ZeroSearchOracle, game, states) = zeros(Float32, length(states))
AZ.state_policy(o::ZeroSearchOracle, game, s) = uniform_pair(o.na)
AZ.state_strategy(o::ZeroSearchOracle, game, s) = uniform_pair(o.na)
AZ.batch_state_policy(o::ZeroSearchOracle, game, states) = (
    fill(Float32(inv(o.na[1])), o.na[1], length(states)),
    fill(Float32(inv(o.na[2])), o.na[2], length(states)),
)
AZ.batch_state_strategy(o::ZeroSearchOracle, game, states) =
    AZ.batch_state_policy(o, game, states)
AZ.state_regret(o::ZeroSearchOracle, game, s) =
    (zeros(Float32, o.na[1]), zeros(Float32, o.na[2]))
AZ.batch_state_regret(o::ZeroSearchOracle, game, states) = (
    zeros(Float32, o.na[1], length(states)),
    zeros(Float32, o.na[2], length(states)),
)

# Support-masked regret prior.
#
# RM+ consumes only [R̄̂]₊ and normalizes it, so the transferred information is
# the positive-regret *support* plus the relative magnitudes inside it. A
# squared-error fit of a sparse nonnegative target minimizes loss by spreading a
# little mass over the truly-zero actions — on the SDA test split the fitted head
# puts 2.77 of 3 components above zero against a true 1.73 — and regret matching
# then reads that smeared mass as real probability. This wrapper keeps the
# baseline head's magnitudes but masks them to the hurdle gate's predicted
# support, which is the variant that best recovers the true regret-matching
# direction (see `score_regret_directions.jl`). An empty support transfers
# nothing, so the node falls back to the cold uniform strategy.
struct SupportMaskedRegretOracle{O,G}
    inner::O            # supplies value and average strategy
    gate::NTuple{2,G}   # hurdle gate trunks, one per player
    threshold::Float32
end

function AZ.state_regret(o::SupportMaskedRegretOracle, game::MG, s)
    r = AZ.state_regret(o.inner, game, s)
    x = MarkovGames.convert_s(Vector{Float32}, s, game)
    return ntuple(2) do p
        gate = vec(o.gate[p](reshape(x, :, 1)))
        r[p] .* (gate .> o.threshold)
    end
end

function AZ.batch_state_regret(o::SupportMaskedRegretOracle, game::MG, states)
    r = AZ.batch_state_regret(o.inner, game, states)
    X = reduce(hcat, MarkovGames.convert_s.(Vector{Float32}, states, (game,)))
    return ntuple(p -> r[p] .* (o.gate[p](X) .> o.threshold), 2)
end

AZ.value(o::SupportMaskedRegretOracle, x) = AZ.value(o.inner, x)
AZ.state_value(o::SupportMaskedRegretOracle, game, s) = AZ.state_value(o.inner, game, s)
AZ.batch_state_value(o::SupportMaskedRegretOracle, game, states) =
    AZ.batch_state_value(o.inner, game, states)
AZ.state_policy(o::SupportMaskedRegretOracle, game, s) = AZ.state_strategy(o.inner, game, s)
AZ.state_strategy(o::SupportMaskedRegretOracle, game, s) = AZ.state_strategy(o.inner, game, s)
AZ.batch_state_policy(o::SupportMaskedRegretOracle, game, states) =
    AZ.batch_state_strategy(o.inner, game, states)
AZ.batch_state_strategy(o::SupportMaskedRegretOracle, game, states) =
    AZ.batch_state_strategy(o.inner, game, states)

# ---------------------------------------------------------------------------
# Checkpoint / refit loading
# ---------------------------------------------------------------------------
checkpoint_dir() = joinpath(EXPERIMENT_DIR, "models_$(SEARCH_NAME)")

function checkpoint_iteration(path)
    m = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(m) && error("Unexpected checkpoint name $(path)")
    return parse(Int, m.captures[1])
end

function select_checkpoint(iter_spec::AbstractString="latest")
    dir = checkpoint_dir()
    isdir(dir) || error("Missing checkpoint directory: $(dir)")
    checkpoints = sort(
        filter(p -> endswith(p, ".jld2"), readdir(dir; join=true));
        by=checkpoint_iteration,
    )
    isempty(checkpoints) && error("No checkpoints in $(dir)")
    iter_spec == "latest" && return last(checkpoints)
    iter = parse(Int, iter_spec)
    matches = filter(p -> checkpoint_iteration(p) == iter, checkpoints)
    isempty(matches) && error("No checkpoint for iteration $(iter)")
    return only(matches)
end

function load_checkpoint_oracle(iter_spec::AbstractString="latest")
    oracle_file = joinpath(EXPERIMENT_DIR, "oracle_$(SEARCH_NAME).jld2")
    isfile(oracle_file) || error("Missing oracle architecture file: $(oracle_file)")
    ckpt = select_checkpoint(iter_spec)
    oracle = AZ.load_oracle(oracle_file)
    oracle isa AZ.FittedRegretModel ||
        error("Expected a FittedRegretModel, got $(typeof(oracle))")
    Flux.loadmodel!(oracle, ckpt)
    return oracle, checkpoint_iteration(ckpt), ckpt
end

softplus_refit_output(x) = Flux.softplus.(x)

function with_softplus_output(actor)
    actor isa Chain || error("Expected checkpoint regret actor to be a Flux.Chain")
    return Chain(actor.layers..., softplus_refit_output)
end

function load_regret_refit(online_oracle, path=BASELINE_MODELS)
    isfile(path) || error("Missing fitted regret models: $(path)")
    fitted = JLD2.load(path)
    refit = AZ.FittedRegretModel(
        deepcopy(online_oracle.shared),
        AZ.MultiActor(
            with_softplus_output(deepcopy(online_oracle.regret_head[1])),
            with_softplus_output(deepcopy(online_oracle.regret_head[2])),
        ),
        deepcopy(online_oracle.strategy_head),
        deepcopy(online_oracle.critic);
        value_weight=online_oracle.value_weight,
        regret_weight=online_oracle.regret_weight,
        strategy_weight=online_oracle.strategy_weight,
    )
    Flux.loadmodel!(refit.regret_head[1], fitted["baseline_p1_state"])
    Flux.loadmodel!(refit.regret_head[2], fitted["baseline_p2_state"])
    return refit
end

# Rebuild the hurdle gate head (trunk + gate layer + sigmoid) for each player.
struct HurdleRegressor{T,G,M}
    trunk::T
    gate::G
    log_magnitude::M
end

Flux.@layer :expand HurdleRegressor

function HurdleRegressor(input_dim::Integer, width::Integer, output_dim::Integer)
    trunk = Chain(
        Dense(input_dim => width, tanh),
        Dense(width => width, tanh),
        Dense(width => width, tanh),
    )
    return HurdleRegressor(trunk, Dense(width => output_dim), Dense(width => output_dim))
end

function load_gate_heads(input_dim::Int, output_dim::Int, path=HURDLE_MODELS)
    isfile(path) || error("Missing hurdle models: $(path)")
    data = JLD2.load(path)
    width = data["metadata"]["width"]
    return ntuple(2) do p
        model = HurdleRegressor(input_dim, width, output_dim)
        Flux.loadmodel!(model, data["hurdle_p$(p)_state"])
        Chain(model.trunk, model.gate, Flux.sigmoid)
    end
end

function load_oracles(game, iter_spec::AbstractString="latest"; gate_threshold=0.5f0)
    online, iter, ckpt = load_checkpoint_oracle(iter_spec)
    refit = load_regret_refit(online)
    input_dim = length(MarkovGames.convert_s(
        Vector{Float32}, rand(core_initialstate_distribution(game)), game,
    ))
    gates = load_gate_heads(input_dim, length(actions(game)[1]))
    masked = SupportMaskedRegretOracle(refit, gates, gate_threshold)
    return (; online, refit, masked, iter, ckpt)
end

# ---------------------------------------------------------------------------
# Solver construction
# ---------------------------------------------------------------------------
rm_style(cfg) = AZ.RegretMatchingSearch(;
    backup=cfg.backup, method=AZ.Plus(), update=get(cfg, :update, :sampled),
)

function build_solver(name::AbstractString, game, oracles, cfg)
    common = (;
        tree_queries=cfg.tree_queries,
        max_depth=cfg.max_depth,
        search_style=rm_style(cfg),
        value_target=cfg.value_target,
        ϵ=_ -> cfg.epsilon,
    )
    name == "zero_oracle" &&
        return AZ.MCTSSearch(; oracle=ZeroSearchOracle(game), common...)
    name == "value_oracle" && return AZ.MCTSSearch(;
        oracle=ValueOnlySearchOracle(game, oracles.refit), common...,
    )
    transfer(oracle) = (;
        oracle,
        prior_scale=cfg.prior_scale,
        regret_prior_weight=1.0,
        statistic_prior_weight=0.0,
        prior_reach_power=cfg.prior_reach_power,
        common...,
    )
    # Current deployment: raw fitted regret added into the node accumulators.
    name == "warmstart" && return AZ.MCTSSearch(;
        transfer(oracles.refit)..., strategy_prior_weight=0.0, transfer_mode=:warmstart,
    )
    # Same prior, read-time mass cap only.
    name == "capped" && return AZ.MCTSSearch(;
        transfer(oracles.refit)..., strategy_prior_weight=0.0,
        transfer_mode=:capped, transfer_cap_ratio=cfg.cap_ratio,
    )
    # Same prior, cap and evidence gate.
    name == "gated" && return AZ.MCTSSearch(;
        transfer(oracles.refit)..., strategy_prior_weight=0.0,
        transfer_mode=:gated, transfer_cap_ratio=cfg.cap_ratio,
        transfer_gate_tol=cfg.gate_tol,
    )
    # Support-masked prior with the current (uncapped) usage, to separate the
    # prior-quality change from the usage change.
    name == "masked_warmstart" && return AZ.MCTSSearch(;
        transfer(oracles.masked)..., strategy_prior_weight=0.0,
        transfer_mode=:warmstart,
    )
    # Both changes.
    name == "masked_gated" && return AZ.MCTSSearch(;
        transfer(oracles.masked)..., strategy_prior_weight=0.0,
        transfer_mode=:gated, transfer_cap_ratio=cfg.cap_ratio,
        transfer_gate_tol=cfg.gate_tol,
    )
    # Theory-consistent averaging: also credit the average strategy with the
    # same gated mass (the object Lemma "approximate Nash after imperfect
    # transfer" actually bounds).
    name == "masked_gated_lemma" && return AZ.MCTSSearch(;
        transfer(oracles.masked)..., strategy_prior_weight=1.0,
        transfer_mode=:gated, transfer_cap_ratio=cfg.cap_ratio,
        transfer_gate_tol=cfg.gate_tol,
    )

    # Uniform-tempered warm start. Spec grammar:
    #   [masked_]temper<ν>          tempering weight ν, reach inside λ
    #   [masked_]temperflat<ν>      same but prior_reach_power = 0, so λ ignores
    #                               reach and tempers on magnitude alone
    #   [masked_]depth<d>           untempered warm start only for depth ≤ d
    # `p` stands in for the decimal point (temper0p1 == ν = 0.1).
    masked = startswith(name, "masked_")
    stem = masked ? name[8:end] : name
    oracle = masked ? oracles.masked : oracles.refit
    m = match(r"^temper(flat)?([0-9p]+)$", stem)
    if !isnothing(m)
        ν = parse(Float64, replace(m.captures[2], "p" => "."))
        reach_power = isnothing(m.captures[1]) ? cfg.prior_reach_power : 0.0
        return AZ.MCTSSearch(;
            oracle,
            prior_scale=cfg.prior_scale,
            regret_prior_weight=1.0,
            strategy_prior_weight=0.0,
            statistic_prior_weight=0.0,
            prior_reach_power=reach_power,
            transfer_mode=:tempered,
            transfer_temper=ν,
            common...,
        )
    end
    m = match(r"^depth(\d+)$", stem)
    if !isnothing(m)
        return AZ.MCTSSearch(;
            transfer(oracle)..., strategy_prior_weight=0.0,
            transfer_mode=:warmstart,
            transfer_max_depth=parse(Int, m.captures[1]),
        )
    end
    error("Unknown solver $(name)")
end

default_config(;
        tree_queries=100, max_depth=5, epsilon=0.1, prior_scale=5.0,
        update::Symbol=:sampled,
    ) = (;
    backup=:mean,
    update,
    value_target=:search,
    tree_queries,
    max_depth,
    epsilon,
    prior_scale,
    prior_reach_power=1.0,
    cap_ratio=0.5,
    gate_tol=1.0,
)

# ---------------------------------------------------------------------------
# Seed-averaged mixed strategy of a search at a state. A best responder cannot
# observe the search's random seed, so the policy it faces is the average over
# search randomness, not one realization.
# ---------------------------------------------------------------------------
function mean_policy(search, game, s, nreps::Int, ϵ)
    na1, na2 = length.(actions(game))
    x = zeros(Float64, na1)
    y = zeros(Float64, na2)
    for _ in 1:nreps
        xi, yi, _ = AZ.search(search, game, s; ϵ)
        x .+= xi
        y .+= yi
    end
    return x ./ nreps, y ./ nreps
end

"""
    truncated_gap(game, search, s0; horizon, nreps, ϵ)

Exact saddle gap of the induced Markov profile on the `horizon`-step truncated
game rooted at `s0`.

Both best responses are computed by full enumeration of the depth-`horizon`
game tree (|A₁|·|A₂| children per node), so this is an exact best response
against the seed-averaged search policy within the truncation — no policy-
gradient approximation and no response-fitting failure mode. Truncation makes it
a lower bound on the untruncated gap, applied identically to every solver.
"""
function truncated_gap(game, search, s0; horizon::Int, nreps::Int=2, ϵ=0.1)
    A1, A2 = actions(game)
    γ = discount(game)

    # value of both sides following the search policy
    function self_value(s, d)
        (d >= horizon || isterminal(game, s)) && return 0.0
        x, y = mean_policy(search, game, s, nreps, ϵ)
        total = 0.0
        for (i, a1) in enumerate(A1), (j, a2) in enumerate(A2)
            p = x[i] * y[j]
            p > 0 || continue
            sp, r = @gen(:sp, :r)(game, s, (a1, a2))
            total += p * (AZ.zs_reward_scalar(r) + γ * self_value(sp, d + 1))
        end
        return total
    end

    # best response value for `player` against the search policy of the other
    function br_value(s, d, player::Int)
        (d >= horizon || isterminal(game, s)) && return 0.0
        x, y = mean_policy(search, game, s, nreps, ϵ)
        own = player == 1 ? A1 : A2
        opp_probs = player == 1 ? y : x
        best = player == 1 ? -Inf : Inf
        for (k, a) in enumerate(own)
            acc = 0.0
            for (l, b) in enumerate(player == 1 ? A2 : A1)
                p = opp_probs[l]
                p > 0 || continue
                joint = player == 1 ? (a, b) : (b, a)
                sp, r = @gen(:sp, :r)(game, s, joint)
                acc += p * (AZ.zs_reward_scalar(r) + γ * br_value(sp, d + 1, player))
            end
            best = player == 1 ? max(best, acc) : min(best, acc)
        end
        return best
    end

    self = self_value(s0, 0)
    b1 = br_value(s0, 0, 1)
    b2 = br_value(s0, 0, 2)
    return (; gap=b1 - b2, gain1=b1 - self, gain2=self - b2, self, b1, b2)
end

# ---------------------------------------------------------------------------
# Head-to-head cross-play
# ---------------------------------------------------------------------------
function play_episode(game, p1_search, p2_search, s; max_steps::Int, ϵ)
    A1, A2 = actions(game)
    γ = discount(game)
    total = 0.0
    disc = 1.0
    for _ in 1:max_steps
        isterminal(game, s) && break
        x, _, _ = AZ.search(p1_search, game, s; ϵ)
        _, y, _ = AZ.search(p2_search, game, s; ϵ)
        idx = AZ.action_idx_from_probs(x, y)
        i, j = Tuple(idx)
        sp, r = @gen(:sp, :r)(game, s, (A1[i], A2[j]))
        total += disc * AZ.zs_reward_scalar(r)
        disc *= γ
        s = sp
    end
    return total
end

# ---------------------------------------------------------------------------
# Fixed exploiter pool.
#
# The PPO responses already trained against `zero_oracle`, `value_oracle`, and
# `full_solver` are reused as a *fixed* pool of adversaries. This avoids the
# failure mode the notes flag for the per-solver PPO diagnostic — a response that
# underfits makes its target look artificially strong — because every solver
# faces exactly the same adversaries. The pool is adversarial by construction:
# each member was trained to exploit one of the existing solvers.
# ---------------------------------------------------------------------------
const PPO_POOL_DIR =
    joinpath(EXPERIMENT_DIR, "ppo_solver_response_utility_results_regret_only")
const PPO_POOL_SOLVERS = ("zero_oracle", "value_oracle", "full_solver")

function load_ppo_actor(trained_against::AbstractString, response_player::Int)
    path = joinpath(
        PPO_POOL_DIR, trained_against, "p$(response_player)",
        "ppo_response_actor_critic.jld2",
    )
    isfile(path) || return nothing
    data = JLD2.load(path)
    meta = data["metadata"]
    get(meta, "solver", nothing) == trained_against ||
        error("PPO model solver mismatch in $(path)")
    get(meta, "response_player", nothing) == response_player ||
        error("PPO model player mismatch in $(path)")
    return data["actor"]
end

actor_probs(actor, game, s) =
    Float64.(vec(Flux.softmax(actor(MarkovGames.convert_s(Vector{Float32}, s, game)))))

# One episode of `search` in seat `tree_player` against a PPO actor in the other.
# Returns the payoff from the tree player's perspective.
function play_vs_actor(game, search, actor, tree_player::Int, s; max_steps::Int, ϵ)
    A1, A2 = actions(game)
    γ = discount(game)
    total = 0.0
    disc = 1.0
    for _ in 1:max_steps
        isterminal(game, s) && break
        x, y, _ = AZ.search(search, game, s; ϵ)
        if tree_player == 1
            p1 = x
            p2 = actor_probs(actor, game, s)
        else
            p1 = actor_probs(actor, game, s)
            p2 = y
        end
        idx = AZ.action_idx_from_probs(p1, p2)
        i, j = Tuple(idx)
        sp, r = @gen(:sp, :r)(game, s, (A1[i], A2[j]))
        payoff = AZ.zs_reward_scalar(r)
        total += disc * (tree_player == 1 ? payoff : -payoff)
        disc *= γ
        s = sp
    end
    return total
end

"""
    exploiter_pool_utility(game, search, states; max_steps, ϵ, seed)

Utility of `search` against every member of the fixed PPO exploiter pool, from
the searching player's perspective in both seats. Higher is better (less
exploited). Returns the per-member seat-summed utilities plus the pool mean and
the pool worst case.
"""
# Every solver sees the same initial states and the same RNG stream per cell, so
# per-episode values are paired across solvers and differences can be tested
# with far less noise than the raw levels.
function exploiter_pool_utility(game, search, states; max_steps::Int=50, ϵ=0.1, seed::Int=0)
    # Keep the two seats separate. A Nash profile in a zero-sum game is one where
    # *each* player's policy is optimal against a best response, so the per-seat
    # best-response utilities are the quantities that have to improve
    # individually: summing them gives a NashConv, which can hide a transfer that
    # helps one seat and hurts the other.
    #
    # Both seat numbers are signed from the searching player's perspective, so
    # higher is better in both. Seat 1 is the value the solver *secures* as
    # player 1 against a best-responding player 2; seat 2 is the negated value it
    # *concedes* as player 2 against a best-responding player 1.
    seat_episodes = Dict{Tuple{String,Int},Vector{Float64}}()
    for member in PPO_POOL_SOLVERS, tree_player in (1, 2)
        # a p1-trained response occupies seat 1, so the tree player is seat 2
        actor = load_ppo_actor(member, tree_player == 1 ? 2 : 1)
        isnothing(actor) && continue
        Random.seed!(seed + 101 * tree_player)
        seat_episodes[(member, tree_player)] = [
            play_vs_actor(game, search, actor, tree_player, s; max_steps, ϵ)
            for s in states
        ]
    end
    isempty(seat_episodes) && return (;
        seat_episodes, seat_pooled=Dict{Int,Vector{Float64}}(), per_member=Dict{String,Float64}(),
        per_seat=Dict{Int,Float64}(), per_seat_worst=Dict{Int,Float64}(),
        pooled=Float64[], pool_mean=NaN, pool_worst=NaN,
    )
    # Per-seat pooled series (mean over pool members, per episode) so seat-level
    # comparisons across solvers are paired.
    seat_pooled = Dict{Int,Vector{Float64}}()
    per_seat = Dict{Int,Float64}()
    per_seat_worst = Dict{Int,Float64}()
    for tree_player in (1, 2)
        series = [v for ((_, p), v) in seat_episodes if p == tree_player]
        isempty(series) && continue
        seat_pooled[tree_player] = reduce(.+, series) ./ length(series)
        per_seat[tree_player] = mean(seat_pooled[tree_player])
        per_seat_worst[tree_player] = minimum(mean.(series))
    end
    # Seat-summed view, retained for comparability with the NashConv-style
    # `summed_ppo_response_utility` in `ppo_solver_response_utilities.jl`.
    per_member = Dict{String,Float64}()
    member_series = Dict{String,Vector{Float64}}()
    for member in PPO_POOL_SOLVERS
        s1 = get(seat_episodes, (member, 1), nothing)
        s2 = get(seat_episodes, (member, 2), nothing)
        (isnothing(s1) || isnothing(s2)) && continue
        member_series[member] = s1 .+ s2
        per_member[member] = mean(member_series[member])
    end
    values = collect(Base.values(per_member))
    pooled = isempty(member_series) ? Float64[] :
        reduce(.+, Base.values(member_series)) ./ length(member_series)
    return (;
        seat_episodes, seat_pooled, per_member, per_seat, per_seat_worst, pooled,
        pool_mean = isempty(values) ? NaN : mean(values),
        pool_worst = isempty(values) ? NaN : minimum(values),
    )
end

paired_delta(a::AbstractVector, b::AbstractVector) = begin
    d = a .- b
    (; delta=mean(d), se=std(d) / sqrt(length(d)))
end

"""
    seat_balanced(game, A, B, states; max_steps, ϵ, seed)

Seat-balanced utility of solver `A` against solver `B`: the mean of A's payoff
in seat 1 and the negation of A's payoff in seat 2, so a positive value favors
`A` regardless of which seat carries the game's asymmetry.
"""
function seat_balanced(game, A, B, states; max_steps::Int=50, ϵ=0.1, seed::Int=0)
    Random.seed!(seed)
    a1 = [play_episode(game, A, B, s; max_steps, ϵ) for s in states]
    Random.seed!(seed)
    a2 = [play_episode(game, B, A, s; max_steps, ϵ) for s in states]
    n = length(states)
    return (;
        balanced=(mean(a1) - mean(a2)) / 2,
        se=sqrt(var(a1) / n + var(a2) / n) / 2,
        as_p1=mean(a1),
        as_p2=mean(a2),
    )
end
