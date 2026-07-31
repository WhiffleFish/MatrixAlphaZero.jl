using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

# Network-free Dubin regret-transfer demonstration.
#
# A finite Dubin tree is solved backward to obtain real leaf values. Every
# nonterminal node then roots an independent full-remaining-depth RM+ MCTS
# search. Ordinary search uses real values for T2 updates. Transfer runs T1=T2
# updates with perturbed values, corrupts its average regret/strategy tables,
# and warm-starts another T2-update search using the real values.

using DelimitedFiles
using LaTeXStrings
using LinearAlgebra
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using POSGModels.Dubin
using POSGModels.StaticArrays
using Plots
using Printf
using ProgressMeter
using Random
using Statistics
using Base.Threads

const AZ = MatrixAlphaZero
const SCRIPT_DIR = @__DIR__
# AAAI uses a 7 in two-column text block with a 0.375 in gutter, hence a
# 3.3125 in = 238.5 bp column. Plots/GR accepts only integer output dimensions,
# so 238 bp is the closest width that cannot overrun \linewidth. GR treats its
# requested "point size" as character height rather than the em size used by
# TeX. With GR's 1.08 conversion and Computer Modern's 0.683 cap-height ratio,
# 6 renders within 0.35 pt of the cap height of AAAI's 10 pt body text.
const AAAI_COLUMN_WIDTH_BP = 238
const AAAI_BODY_PLOTS_POINTSIZE = 6
const AAAI_FIGURE_HEIGHT_BP = 493
const PAPER_ROOT = JointDubinState(
    SA[1.806285497972572, 2.411204504285737, -0.2705116498710156],
    SA[4.870234852067901, 2.5076120564966677, -2.6982903330828334],
)

struct DepthState
    inner::JointDubinState
    depth::Int
    path::Int
end

struct FiniteDubinGame{G} <: MG{DepthState,Tuple{Int,Int}}
    inner::G
    horizon::Int
end

POMDPs.actions(game::FiniteDubinGame) = actions(game.inner)
POMDPs.discount(game::FiniteDubinGame) = discount(game.inner)
POMDPs.initialstate(game::FiniteDubinGame) =
    Deterministic(DepthState(game.inner.initialstate, 0, 0))
POMDPs.isterminal(game::FiniteDubinGame, s::DepthState) =
    isterminal(game.inner, s.inner)

function POMDPs.gen(
        game::FiniteDubinGame,
        s::DepthState,
        action::Tuple{Int,Int},
        rng::AbstractRNG=Random.default_rng(),
    )
    sp, r = @gen(:sp, :r)(game.inner, s.inner, action, rng)
    na1 = length(actions(game)[1])
    joint_index = (action[2] - 1) * na1 + action[1]
    path = s.path * prod(length.(actions(game))) + joint_index
    return (; sp=DepthState(sp, s.depth + 1, path), r)
end

# The tabular oracle decodes only the exact finite-tree address. Float32 is
# exact for the default depth-five tree (largest path < 9^5).
POMDPs.convert_s(::Type{Vector{Float32}}, s::DepthState, ::FiniteDubinGame) =
    Float32[s.depth, s.path]

state_key(s::DepthState) = (s.depth, s.path)

struct TabularOracle
    values::Dict{Tuple{Int,Int},Float64}
    regrets::Dict{Tuple{Int,Int},NTuple{2,Vector{Float64}}}
    strategies::Dict{Tuple{Int,Int},NTuple{2,Vector{Float64}}}
    na::NTuple{2,Int}
end

uniform_pair(oracle::TabularOracle) = (
    fill(inv(oracle.na[1]), oracle.na[1]),
    fill(inv(oracle.na[2]), oracle.na[2]),
)

function AZ.value(oracle::TabularOracle, x::AbstractVector)
    key = (round(Int, x[1]), round(Int, x[2]))
    return Float32[get(oracle.values, key, 0.0)]
end

function AZ.value(oracle::TabularOracle, x::AbstractMatrix)
    values = Float32[get(oracle.values, (round(Int, col[1]), round(Int, col[2])), 0.0)
                     for col in eachcol(x)]
    return reshape(values, 1, :)
end

AZ.state_value(oracle::TabularOracle, game, s::DepthState) =
    get(oracle.values, state_key(s), 0.0)
AZ.batch_state_value(oracle::TabularOracle, game, states) =
    [get(oracle.values, state_key(s), 0.0) for s in states]
AZ.state_regret(oracle::TabularOracle, game, s::DepthState) =
    get(oracle.regrets, state_key(s), (zeros(oracle.na[1]), zeros(oracle.na[2])))
AZ.state_strategy(oracle::TabularOracle, game, s::DepthState) =
    get(oracle.strategies, state_key(s), uniform_pair(oracle))
AZ.state_policy(oracle::TabularOracle, game, s::DepthState) =
    AZ.state_strategy(oracle, game, s)
AZ.batch_state_strategy(oracle::TabularOracle, game, states) = (
    reduce(hcat, (AZ.state_strategy(oracle, game, s)[1] for s in states)),
    reduce(hcat, (AZ.state_strategy(oracle, game, s)[2] for s in states)),
)
AZ.batch_state_policy(oracle::TabularOracle, game, states) =
    AZ.batch_state_strategy(oracle, game, states)
AZ.batch_state_regret(oracle::TabularOracle, game, states) = (
    reduce(hcat, (AZ.state_regret(oracle, game, s)[1] for s in states)),
    reduce(hcat, (AZ.state_regret(oracle, game, s)[2] for s in states)),
)

struct FiniteTree
    game::FiniteDubinGame
    states::Vector{DepthState}
    depths::Vector{Int}
    children::Vector{Matrix{Int}}
    rewards::Vector{Matrix{Float64}}
    layers::Vector{Vector{Int}}
    index::Dict{Tuple{Int,Int},Int}
    values::Vector{Float64}
    exact_strategy::Vector{NTuple{2,Vector{Float64}}}
end

function supports(n::Int, size::Int)
    size == 1 && return [[i] for i in 1:n]
    size == 2 && return [[i, j] for i in 1:n for j in (i + 1):n]
    size == n && return [collect(1:n)]
    return Vector{Int}[]
end

function solve_zero_sum(payoff::AbstractMatrix; tolerance::Float64=1e-9)
    n1, n2 = size(payoff)
    for support_size in 1:min(n1, n2)
        for support1 in supports(n1, support_size), support2 in supports(n2, support_size)
            subgame = payoff[support1, support2]
            system2 = [subgame -ones(support_size); ones(1, support_size) 0.0]
            system1 = [transpose(subgame) -ones(support_size); ones(1, support_size) 0.0]
            solution2 = try
                system2 \ vcat(zeros(support_size), 1.0)
            catch error
                error isa LinearAlgebra.SingularException || rethrow()
                continue
            end
            solution1 = try
                system1 \ vcat(zeros(support_size), 1.0)
            catch error
                error isa LinearAlgebra.SingularException || rethrow()
                continue
            end
            x, y = zeros(n1), zeros(n2)
            x[support1] .= solution1[1:support_size]
            y[support2] .= solution2[1:support_size]
            minimum(x) >= -tolerance || continue
            minimum(y) >= -tolerance || continue
            x .= max.(x, 0.0); y .= max.(y, 0.0)
            sum(x) > 0 && sum(y) > 0 || continue
            x ./= sum(x); y ./= sum(y)
            row_values = payoff * y
            column_values = transpose(payoff) * x
            value = 0.5 * (maximum(row_values) + minimum(column_values))
            maximum(row_values) <= value + 20tolerance || continue
            minimum(column_values) >= value - 20tolerance || continue
            return x, y, value
        end
    end
    error("Failed to solve zero-sum matrix:\n$(payoff)")
end

function build_tree(inner_game::DubinMG, root::JointDubinState, horizon::Int)
    game = FiniteDubinGame(
        DubinMG(;
            actions=inner_game.actions,
            V=inner_game.V,
            tag_reward=inner_game.tag_reward,
            discount=inner_game.discount,
            floor=inner_game.floor,
            initialstate=root,
            goal=inner_game.goal,
            dt=inner_game.dt,
        ),
        horizon,
    )
    root_state = DepthState(root, 0, 0)
    states = DepthState[root_state]
    depths = Int[0]
    children = Matrix{Int}[zeros(Int, 0, 0)]
    rewards = Matrix{Float64}[zeros(0, 0)]
    layers = [Int[] for _ in 0:horizon]
    push!(layers[1], 1)
    index = Dict(state_key(root_state) => 1)
    A1, A2 = actions(game)

    for depth in 0:(horizon - 1)
        for node in layers[depth + 1]
            s = states[node]
            isterminal(game, s) && continue
            child_matrix = zeros(Int, length(A1), length(A2))
            reward_matrix = zeros(Float64, length(A1), length(A2))
            for (i, a1) in enumerate(A1), (j, a2) in enumerate(A2)
                sp, r = @gen(:sp, :r)(game, s, (a1, a2))
                push!(states, sp)
                push!(depths, depth + 1)
                push!(children, zeros(Int, 0, 0))
                push!(rewards, zeros(0, 0))
                child = length(states)
                child_matrix[i, j] = child
                reward_matrix[i, j] = AZ.zs_reward_scalar(r)
                index[state_key(sp)] = child
                push!(layers[depth + 2], child)
            end
            children[node] = child_matrix
            rewards[node] = reward_matrix
        end
    end

    values = zeros(Float64, length(states))
    exact_strategy = [(fill(inv(length(A1)), length(A1)), fill(inv(length(A2)), length(A2)))
                      for _ in states]
    gamma = discount(game)

    for depth in (horizon - 1):-1:0
        layer = layers[depth + 1]
        @threads for k in eachindex(layer)
            node = layer[k]
            isempty(children[node]) && continue
            q = rewards[node] .+ gamma .* values[children[node]]
            x, y, value = solve_zero_sum(q)
            values[node] = value
            exact_strategy[node] = (x, y)
        end
    end
    return FiniteTree(
        game, states, depths, children, rewards, layers, index,
        values, exact_strategy,
    )
end

function parse_list(::Type{T}, value::AbstractString) where T
    result = parse.(T, filter(!isempty, strip.(split(value, ','))))
    isempty(result) && error("Expected a nonempty comma-separated list")
    return unique(result)
end

function parse_cli(args)
    cfg = Dict{String,Any}(
        "depth" => 5,
        "iterations" => 64,
        "trials" => 32,
        "leaf_errors" => [0.0, 0.20, 0.40, 0.60, 0.80, 1.00],
        "transfer_errors" => [0.0, 0.025, 0.05, 0.10, 0.20, 0.40],
        "epsilon" => 0.10,
        "seed" => 20260719,
        "output" => joinpath(SCRIPT_DIR, "tabular_regret_transfer_results"),
        "test" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test"
            cfg["test"] = true
            i += 1
            continue
        end
        startswith(arg, "--") || error("Unexpected argument $(arg)")
        parts = split(arg[3:end], '='; limit=2)
        key = replace(first(parts), '-' => '_')
        haskey(cfg, key) || error("Unknown option --$(first(parts))")
        value = if length(parts) == 2
            last(parts)
        else
            i += 1
            i <= length(args) || error("Missing value for $(arg)")
            args[i]
        end
        if key in ("output",)
            cfg[key] = value
        elseif key in ("epsilon",)
            cfg[key] = parse(Float64, value)
        elseif key in ("leaf_errors", "transfer_errors")
            cfg[key] = parse_list(Float64, value)
        elseif key != "test"
            cfg[key] = parse(Int, value)
        end
        i += 1
    end
    if cfg["test"]
        merge!(cfg, Dict(
            "depth" => 2,
            "iterations" => 8,
            "trials" => 2,
            "leaf_errors" => [0.0, 0.2],
            "transfer_errors" => [0.0, 0.2],
            "output" => joinpath(abspath(cfg["output"]), "smoke"),
        ))
    end
    cfg["depth"] > 0 || error("depth must be positive")
    cfg["iterations"] > 1 || error("iterations must exceed one")
    cfg["trials"] > 0 || error("trials must be positive")
    all(e -> 0 <= e <= 1.0, cfg["leaf_errors"]) || error("leaf errors must be in [0, 1]")
    all(e -> 0 <= e <= 0.5, cfg["transfer_errors"]) || error("transfer errors must be in [0, 0.5]")
    0 <= cfg["epsilon"] <= 1 || error("epsilon must be in [0, 1]")
    return cfg
end

function exact_value_dict(tree::FiniteTree)
    return Dict(state_key(s) => tree.values[i] for (i, s) in enumerate(tree.states))
end

function empty_prior_dicts(tree::FiniteTree)
    na = Tuple(length.(actions(tree.game)))
    regrets = Dict{Tuple{Int,Int},NTuple{2,Vector{Float64}}}()
    strategies = Dict(state_key(s) => (fill(inv(na[1]), na[1]), fill(inv(na[2]), na[2]))
                      for s in tree.states)
    return regrets, strategies
end

function corrupted_values(tree::FiniteTree, error::Float64, seed::Int)
    rng = MersenneTwister(seed)
    noise = error .* (2 .* rand(rng, length(tree.states)) .- 1)
    for i in eachindex(noise)
        isterminal(tree.game, tree.states[i]) && (noise[i] = 0.0)
    end
    values = Dict(state_key(s) => tree.values[i] + noise[i] for (i, s) in enumerate(tree.states))
    achieved = maximum(abs(values[state_key(s)] - tree.values[i])
                       for (i, s) in enumerate(tree.states))
    return values, achieved
end

function regret_perturbation(rng, target, error)
    iszero(error) && return copy(target)
    return target .+ error .* (2 .* rand(rng, length(target)) .- 1)
end

function strategy_perturbation(rng, target, error)
    iszero(error) && return copy(target)
    length(target) == 1 && return copy(target)
    # Rejection sampling in the first n-1 coordinates is uniform on the
    # intersection of the probability simplex and the L-infinity error box.
    while true
        delta = zeros(length(target))
        delta[1:(end - 1)] .= error .* (2 .* rand(rng, length(target) - 1) .- 1)
        delta[end] = -sum(@view delta[1:(end - 1)])
        abs(delta[end]) <= error || continue
        candidate = target .+ delta
        all(x -> 0.0 <= x <= 1.0, candidate) || continue
        return candidate
    end
end

function corrupted_priors(
        tree::FiniteTree,
        source_regret,
        source_strategy,
        error::Float64,
        seed::Int,
    )
    rng = MersenneTwister(seed)
    regrets = Dict{Tuple{Int,Int},NTuple{2,Vector{Float64}}}()
    strategies = Dict{Tuple{Int,Int},NTuple{2,Vector{Float64}}}()
    max_regret_error = 0.0
    max_strategy_error = 0.0
    for i in eachindex(tree.states)
        s = tree.states[i]
        key = state_key(s)
        if isempty(tree.children[i])
            strategies[key] = source_strategy[i]
            continue
        end
        corrupted_r = ntuple(2) do player
            exact = source_regret[i][player]
            predicted_average = regret_perturbation(rng, exact, error)
            max_regret_error = max(max_regret_error, maximum(abs.(predicted_average .- exact)))
            # Current MCTSSearch multiplies average regret directly by
            # prior_scale=T1 to reconstruct the transferred cumulative regret.
            predicted_average
        end
        corrupted_s = ntuple(2) do player
            exact = source_strategy[i][player]
            predicted = strategy_perturbation(rng, exact, error)
            max_strategy_error = max(max_strategy_error, maximum(abs.(predicted .- exact)))
            predicted
        end
        regrets[key] = corrupted_r
        strategies[key] = corrupted_s
    end
    return regrets, strategies, max_regret_error, max_strategy_error
end

function tabular_oracle(tree::FiniteTree, values; regrets=nothing, strategies=nothing)
    empty_regrets, empty_strategies = empty_prior_dicts(tree)
    return TabularOracle(
        values,
        isnothing(regrets) ? empty_regrets : regrets,
        isnothing(strategies) ? empty_strategies : strategies,
        Tuple(length.(actions(tree.game))),
    )
end

function run_node_search(
        tree::FiniteTree,
        oracle::TabularOracle,
        cfg,
        node::Int,
        seed::Int;
        transfer::Bool,
    )
    iterations = cfg["iterations"]
    remaining_depth = tree.game.horizon - tree.depths[node]
    remaining_depth > 0 || error("Cannot search finite-tree leaf node $(node)")
    search = AZ.MCTSSearch(;
        oracle,
        # The first query expands the root without a regret update. The extra
        # query makes the number of local RM+ updates exactly T1 or T2.
        tree_queries=iterations + 1,
        max_depth=remaining_depth,
        search_style=AZ.RegretMatchingSearch(; backup=:sample, method=AZ.Plus()),
        ϵ=_ -> cfg["epsilon"],
        prior_scale=transfer ? Float64(iterations) : 0.0,
    )
    Random.seed!(seed)
    (_, _, _), info = AZ.search_info(search, tree.game, tree.states[node]; ϵ=cfg["epsilon"])
    prior_updates = transfer ? iterations : 0
    updates = sum(info.tree.n_sa[1]) - prior_updates
    isapprox(updates, iterations; atol=1e-8, rtol=0.0) ||
        error("Expected $(iterations) root updates, observed $(updates)")
    return info.tree
end

function solve_all_nodes(
        tree::FiniteTree,
        oracle::TabularOracle,
        cfg,
        seed::Int;
        transfer::Bool,
        extract_source::Bool=false,
    )
    na = Tuple(length.(actions(tree.game)))
    policy1 = [fill(inv(na[1]), na[1]) for _ in tree.states]
    policy2 = [fill(inv(na[2]), na[2]) for _ in tree.states]
    average_regret = [(zeros(na[1]), zeros(na[2])) for _ in tree.states]
    average_strategy = [(copy(policy1[i]), copy(policy2[i])) for i in eachindex(tree.states)]
    nodes = findall(i -> !isempty(tree.children[i]), eachindex(tree.states))
    iterations = cfg["iterations"]
    @threads for k in eachindex(nodes)
        node = nodes[k]
        search_tree = run_node_search(
            tree, oracle, cfg, node, seed + 104729node; transfer,
        )
        policy1[node] = AZ.normalized_or_uniform(search_tree.policy_sum[1][1])
        policy2[node] = AZ.normalized_or_uniform(search_tree.policy_sum[2][1])
        if extract_source
            average_regret[node] = (
                search_tree.regret[1][1] ./ iterations,
                search_tree.regret[2][1] ./ iterations,
            )
            average_strategy[node] = (copy(policy1[node]), copy(policy2[node]))
        end
    end
    return (; policy=(policy1, policy2), average_regret, average_strategy)
end

function exact_nash_gap(tree::FiniteTree, policy1, policy2)
    lower = zeros(length(tree.states))
    upper = zeros(length(tree.states))
    gamma = discount(tree.game)
    for depth in (tree.game.horizon - 1):-1:0
        for node in tree.layers[depth + 1]
            isempty(tree.children[node]) && continue
            q_lower = tree.rewards[node] .+ gamma .* lower[tree.children[node]]
            q_upper = tree.rewards[node] .+ gamma .* upper[tree.children[node]]
            lower[node] = minimum(transpose(policy1[node]) * q_lower)
            upper[node] = maximum(q_upper * policy2[node])
        end
    end
    return (; lower=lower[1], upper=upper[1], gap=max(upper[1] - lower[1], 0.0))
end

function evaluate_baseline(tree, cfg, trial)
    trial_seed = cfg["seed"] + 1_000_000 * trial
    oracle = tabular_oracle(tree, exact_value_dict(tree))
    solved = solve_all_nodes(
        tree, oracle, cfg, trial_seed + 10; transfer=false,
    )
    baseline = exact_nash_gap(tree, solved.policy...)
    return (; gap=baseline.gap, achieved_leaf_error=0.0)
end

function evaluate_source(tree, cfg, leaf_error, trial)
    trial_seed = cfg["seed"] + 1_000_000 * trial
    values, achieved_leaf = corrupted_values(tree, leaf_error, trial_seed + 1)
    oracle = tabular_oracle(tree, values)
    solved = solve_all_nodes(
        tree, oracle, cfg, trial_seed + 20; transfer=false, extract_source=true,
    )
    return (; solved.average_regret, solved.average_strategy, achieved_leaf_error=achieved_leaf)
end

function evaluate_transfer(tree, cfg, source, transfer_error, trial)
    trial_seed = cfg["seed"] + 1_000_000 * trial
    regrets, strategies, achieved_regret, achieved_strategy = corrupted_priors(
        tree,
        source.average_regret,
        source.average_strategy,
        transfer_error,
        trial_seed + 2,
    )
    oracle = tabular_oracle(tree, exact_value_dict(tree); regrets, strategies)
    solved = solve_all_nodes(
        tree, oracle, cfg, trial_seed + 10; transfer=true,
    )
    transferred = exact_nash_gap(tree, solved.policy...)
    return (;
        gap=transferred.gap,
        achieved_leaf_error=source.achieved_leaf_error,
        achieved_regret_error=achieved_regret,
        achieved_strategy_error=achieved_strategy,
    )
end


function condition_row(leaf_error, transfer_error, trial, baseline, transferred)
    return (;
        leaf_error,
        transfer_error,
        trial,
        baseline_gap=baseline.gap,
        transfer_gap=transferred.gap,
        improvement=baseline.gap - transferred.gap,
        achieved_leaf_error=transferred.achieved_leaf_error,
        achieved_regret_error=transferred.achieved_regret_error,
        achieved_strategy_error=transferred.achieved_strategy_error,
    )
end

function write_csv(path, header, rows)
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] .= header
    for (i, row) in enumerate(rows)
        table[i + 1, :] .= row
    end
    writedlm(path, table, ',')
    return path
end

function sem(values)
    length(values) <= 1 && return 0.0
    return std(values) / sqrt(length(values))
end

function summarize_cells(rows, leaf_errors, transfer_errors)
    summary = NamedTuple[]
    baseline = zeros(length(transfer_errors), length(leaf_errors))
    transferred = similar(baseline)
    improvement = similar(baseline)
    for (ix, leaf_error) in enumerate(leaf_errors), (iy, transfer_error) in enumerate(transfer_errors)
        cell = filter(r -> r.leaf_error == leaf_error && r.transfer_error == transfer_error, rows)
        b = getproperty.(cell, :baseline_gap)
        t = getproperty.(cell, :transfer_gap)
        g = getproperty.(cell, :improvement)
        baseline[iy, ix] = mean(b)
        transferred[iy, ix] = mean(t)
        improvement[iy, ix] = mean(g)
        push!(summary, (;
            leaf_error, transfer_error, trials=length(cell),
            baseline_gap_mean=mean(b), baseline_gap_sem=sem(b),
            transfer_gap_mean=mean(t), transfer_gap_sem=sem(t),
            improvement_mean=mean(g), improvement_sem=sem(g),
            achieved_leaf_error=maximum(getproperty.(cell, :achieved_leaf_error)),
            achieved_regret_error=maximum(getproperty.(cell, :achieved_regret_error)),
            achieved_strategy_error=maximum(getproperty.(cell, :achieved_strategy_error)),
        ))
    end
    return summary, baseline, transferred, improvement
end

function annotate_heatmap!(plot_handle, xs, ys, matrix; text_color="#1A1A1A")
    for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
        annotate!(
            plot_handle,
            x,
            y,
            text(
                @sprintf("%.2f", matrix[iy, ix]);
                family="Computer Modern", pointsize=AAAI_BODY_PLOTS_POINTSIZE,
                color=text_color,
                halign=:center, valign=:center,
            ),
        )
    end
    return plot_handle
end

function add_cell_boundaries!(plot_handle, nx::Int, ny::Int)
    for x in range(0.5, nx + 0.5; length=nx + 1)
        plot!(plot_handle, [x, x], [0.5, ny + 0.5];
              color=:white, alpha=0.75, linewidth=0.6, label="")
    end
    for y in range(0.5, ny + 0.5; length=ny + 1)
        plot!(plot_handle, [0.5, nx + 0.5], [y, y];
              color=:white, alpha=0.75, linewidth=0.6, label="")
    end
    return plot_handle
end

function save_heatmaps(
        output, leaf_errors, transfer_errors, transferred, improvement,
        baseline_mean, baseline_std,
    )
    default(grid=false, framestyle=:box, fontfamily="Computer Modern")
    # The tested errors are a discrete experimental grid.  Plotting their raw
    # numeric values compresses the small-error cells (0, .025, .05), making
    # annotations overlap, so use equally spaced cells with numeric tick labels.
    x_positions = collect(eachindex(leaf_errors))
    y_positions = collect(eachindex(transfer_errors))
    x_labels = [@sprintf("%.3g", value) for value in leaf_errors]
    y_labels = [@sprintf("%.3g", value) for value in transfer_errors]
    axis_ticks = (;
        xticks=(x_positions, x_labels),
        yticks=(y_positions, y_labels),
    )
    common_max = max(baseline_mean + baseline_std, maximum(transferred), eps())
    sequential_palette = cgrad(["#F7FBFF", "#C6DBEF", "#6BAED6"])
    improvement_palette = cgrad(["#D98C82", "#F7F7F7", "#7FB3D1"], [0.0, 0.5, 1.0])
    p_transfer = heatmap(
        x_positions, y_positions, transferred;
        xlabel="",
        ylabel=L"\mathrm{Transfer\ error}\ \delta_R=\delta_\sigma",
        title="(a) Nash gap", color=sequential_palette,
        clims=(0, common_max), colorbar=false,
        xlims=(0.5, length(x_positions) + 0.5),
        ylims=(0.5, length(y_positions) + 0.5),
        axis_ticks...,
    )
    gain_max = max(maximum(abs, improvement), eps())
    p_improvement = heatmap(
        x_positions, y_positions, improvement;
        xlabel=L"\mathrm{Leaf\ value\ error}\ \delta_V",
        ylabel=L"\mathrm{Transfer\ error}\ \delta_R=\delta_\sigma",
        title="(b) Gap reduction",
        color=improvement_palette, clims=(-gain_max, gain_max), colorbar=false,
        xlims=(0.5, length(x_positions) + 0.5),
        ylims=(0.5, length(y_positions) + 0.5),
        axis_ticks...,
    )
    add_cell_boundaries!(p_transfer, length(x_positions), length(y_positions))
    add_cell_boundaries!(p_improvement, length(x_positions), length(y_positions))
    annotate_heatmap!(p_transfer, x_positions, y_positions, transferred)
    annotate_heatmap!(p_improvement, x_positions, y_positions, improvement)
    # Keep the ordinary RM+ reference in the paper caption rather than shrinking
    # the two panels with an additional in-figure heading.
    figure = plot(
        p_transfer, p_improvement;
        layout=(2, 1), size=(AAAI_COLUMN_WIDTH_BP, AAAI_FIGURE_HEIGHT_BP),
        titlefontsize=AAAI_BODY_PLOTS_POINTSIZE,
        guidefontsize=AAAI_BODY_PLOTS_POINTSIZE,
        tickfontsize=AAAI_BODY_PLOTS_POINTSIZE,
        margin=3Plots.mm, left_margin=5Plots.mm, bottom_margin=5Plots.mm,
    )
    savefig(figure, joinpath(@__DIR__, output, "regret_transfer_heatmaps.png"))
    savefig(figure, joinpath(@__DIR__, output, "regret_transfer_heatmaps.pdf"))
    return figure
end

function main()
    cfg = parse_cli(ARGS)
    output = abspath(cfg["output"])
    mkpath(output)
    inner_game = DubinMG(V=(1.0, 1.0))

    println("Network-free Dubin regret-transfer heatmap")
    println("threads=$(nthreads()) depth=$(cfg["depth"]) T1=T2=$(cfg["iterations"])")
    state = PAPER_ROOT
    tree = build_tree(inner_game, state, cfg["depth"])
    @printf(
        "Selected state: attacker=(%.3f, %.3f, %.3f) defender=(%.3f, %.3f, %.3f)\n",
        state.attacker..., state.defender...,
    )

    leaf_errors = sort(Float64.(cfg["leaf_errors"]))
    transfer_errors = sort(Float64.(cfg["transfer_errors"]))
    total = length(leaf_errors) * length(transfer_errors) * cfg["trials"]
    progress = Progress(total; desc="heatmap trials", showspeed=true)
    rows = NamedTuple[]
    baselines = Dict(
        trial => evaluate_baseline(tree, cfg, trial)
        for trial in 1:cfg["trials"]
    )
    for leaf_error in leaf_errors, trial in 1:cfg["trials"]
        # Ordinary RM+ always uses real leaf values and is independent of both
        # error axes. The perturbed source solve is shared across its row.
        baseline = baselines[trial]
        source = evaluate_source(tree, cfg, leaf_error, trial)
        for transfer_error in transfer_errors
            transferred = evaluate_transfer(tree, cfg, source, transfer_error, trial)
            row = condition_row(leaf_error, transfer_error, trial, baseline, transferred)
            push!(rows, row)
            next!(progress; showvalues=[
                (:leaf_error, leaf_error), (:transfer_error, transfer_error),
                (:gain, round(row.improvement; digits=4)),
            ])
        end
    end
    finish!(progress)

    trial_header = String.(propertynames(first(rows)))
    trial_rows = [Any[getproperty(row, Symbol(name)) for name in trial_header] for row in rows]
    write_csv(joinpath(@__DIR__, output, "trials.csv"), trial_header, trial_rows)
    summary, _, transferred, improvement = summarize_cells(rows, leaf_errors, transfer_errors)
    summary_header = String.(propertynames(first(summary)))
    summary_rows = [Any[getproperty(row, Symbol(name)) for name in summary_header] for row in summary]
    write_csv(joinpath(@__DIR__, output, "summary.csv"), summary_header, summary_rows)

    ordinary_gaps = [baselines[trial].gap for trial in 1:cfg["trials"]]
    ordinary_mean = mean(ordinary_gaps)
    ordinary_std = std(ordinary_gaps)
    ordinary_sem = sem(ordinary_gaps)
    write_csv(
        joinpath(@__DIR__, output, "ordinary_rm_plus_summary.csv"),
        ["evaluations", "mean_nash_gap", "std_nash_gap", "sem_nash_gap"],
        [Any[length(ordinary_gaps), ordinary_mean, ordinary_std, ordinary_sem]],
    )
    save_heatmaps(
        output, leaf_errors, transfer_errors, transferred, improvement,
        ordinary_mean, ordinary_std,
    )

    open(joinpath(@__DIR__, output, "config.txt"), "w") do io
        for key in sort!(collect(keys(cfg)))
            println(io, key, "=", cfg[key])
        end
        println(io, "threads=", nthreads())
        println(io, "selected_attacker=", collect(state.attacker))
        println(io, "selected_defender=", collect(state.defender))
        println(io, "finite_tree_nodes=", length(tree.states))
    end
    println("Wrote results to $(output)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
