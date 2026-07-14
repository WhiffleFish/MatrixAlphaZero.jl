using Pkg
Pkg.activate("experiments")

# Compare ordinary regret matching with regret matching+ at one fixed Dubin
# state. RM+ uses the same sampled updates as RM, but truncates each updated
# cumulative-regret vector at zero before the next tree query.
#
# Example:
#   julia --project=experiments experiments/dubin/rm_plus_regret_diagnostics.jl \
#       --runs 100 --queries 500,1000,5000 --max_depth 5

using DelimitedFiles
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POSGModels.Dubin
using POSGModels.StaticArrays
using Printf
using Random
using Statistics

const AZ = MatrixAlphaZero

struct ZeroSearchOracle
    na::NTuple{2,Int}
end

uniform_pair(oracle::ZeroSearchOracle) =
    (fill(1.0f0 / oracle.na[1], oracle.na[1]), fill(1.0f0 / oracle.na[2], oracle.na[2]))

AZ.state_value(::ZeroSearchOracle, game, s) = 0.0
AZ.batch_state_value(oracle::ZeroSearchOracle, game, states) = fill(0.0, length(states))
AZ.value(::ZeroSearchOracle, x::AbstractVector) = Float32[0.0]
AZ.value(::ZeroSearchOracle, x::AbstractMatrix) = zeros(Float32, 1, size(x, 2))
AZ.state_policy(oracle::ZeroSearchOracle, game, s) = uniform_pair(oracle)
AZ.batch_state_policy(oracle::ZeroSearchOracle, game, states) = (
    fill(1.0f0 / oracle.na[1], oracle.na[1], length(states)),
    fill(1.0f0 / oracle.na[2], oracle.na[2], length(states)),
)

function parse_queries(value::AbstractString)
    queries = parse.(Int, split(value, ','))
    isempty(queries) && error("--queries must contain at least one query budget")
    all(>(0), queries) || error("all --queries values must be positive")
    return unique(queries)
end

function parse_cli(args)
    cfg = Dict{String,Any}(
        "runs" => 100,
        "queries" => [500, 1000, 5000],
        "max_depth" => 5,
        "epsilon" => 0.30,
        # Matches the regret-matching seed sequence in search_solver_diagnostics.jl.
        "seed" => 10_000,
        "backup" => "sample",
        "output" => joinpath(@__DIR__, "rm_plus_regret_diagnostics_results"),
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test"
            cfg["runs"] = 2
            cfg["queries"] = [4]
            cfg["max_depth"] = 2
        elseif startswith(arg, "--")
            key_value = split(arg[3:end], '='; limit=2)
            key = first(key_value)
            value = if length(key_value) == 2
                last(key_value)
            else
                i < length(args) || error("Missing value for --$(key)")
                i += 1
                args[i]
            end

            if key in ("backup", "output")
                cfg[key] = value
            elseif key in ("runs", "max_depth", "seed")
                cfg[key] = parse(Int, value)
            elseif key in ("queries", "tree_queries")
                cfg["queries"] = parse_queries(value)
            elseif key == "epsilon"
                cfg[key] = parse(Float64, value)
            else
                error("Unknown argument --$(key)")
            end
        else
            error("Unexpected positional argument $(arg)")
        end
        i += 1
    end

    cfg["runs"] > 0 || error("--runs must be positive")
    cfg["max_depth"] > 0 || error("--max_depth must be positive")
    0.0 <= cfg["epsilon"] <= 1.0 || error("--epsilon must be in [0, 1]")
    return cfg
end

dubin_reference_state() =
    JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])

function run_trial(game, oracle, s0; queries::Int, max_depth::Int, epsilon::Float64, backup::Symbol, rm_plus::Bool)
    method = rm_plus ? AZ.Plus() : AZ.Vanilla()
    style = AZ.RegretMatchingSearch(; backup, method)
    search = AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth,
        ϵ=_ -> epsilon,
        search_style=style,
    )
    tree = AZ.Tree(search, game, s0)
    elapsed = @elapsed for _ in 1:queries
        AZ.simulate(search, tree, game, 1; ϵ=epsilon)
    end

    visits = tree.n_s[1]
    denom_sqrt = sqrt(max(visits, 1))
    denom_mean = max(visits, 1)
    raw = (copy(tree.regret[1][1]), copy(tree.regret[2][1]))
    normalized = (raw[1] ./ denom_sqrt, raw[2] ./ denom_sqrt)
    average = (raw[1] ./ denom_mean, raw[2] ./ denom_mean)
    strategy = (
        AZ.normalized_or_uniform(tree.policy_sum[1][1]),
        AZ.normalized_or_uniform(tree.policy_sum[2][1]),
    )
    return (; raw, normalized, average, strategy, visits, elapsed)
end

function regret_metrics(regret::AbstractVector)
    positive = max.(regret, 0.0)
    negative = max.(-regret, 0.0)
    positive_mass = sum(positive)
    negative_mass = sum(negative)
    positive_probs = positive_mass > 0 ? positive ./ positive_mass : fill(NaN, length(regret))
    hhi = positive_mass > 0 ? sum(abs2, positive_probs) : NaN
    entropy = positive_mass > 0 ? -sum(p > 0 ? p * log(p) : 0.0 for p in positive_probs) : NaN
    return (;
        positive_l1=positive_mass,
        positive_l2=sqrt(sum(abs2, positive)),
        negative_l1=negative_mass,
        negative_l2=sqrt(sum(abs2, negative)),
        signed_l2=sqrt(sum(abs2, regret)),
        positive_hhi=hhi,
        positive_entropy=entropy,
        positive_support=count(>(0.0), positive),
        dominant_action=argmax(regret),
        minimum_regret=minimum(regret),
        maximum_regret=maximum(regret),
    )
end

function write_csv(path::AbstractString, header, rows)
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] .= header
    for (i, row) in enumerate(rows)
        table[i + 1, :] .= row
    end
    writedlm(path, table, ',')
    return path
end

function write_matrix(path::AbstractString, matrix)
    header = ["action_$(i)" for i in axes(matrix, 2)]
    rows = [collect(row) for row in eachrow(matrix)]
    return write_csv(path, header, rows)
end

function result_matrix(results, field::Symbol, player::Int)
    return reduce(vcat, [permutedims(getproperty(result, field)[player]) for result in results])
end

function finite_mean(values)
    finite_values = filter(isfinite, values)
    return isempty(finite_values) ? NaN : mean(finite_values)
end

function summarize_group(queries, solver, player, results)
    metrics = regret_metrics.(getindex.(getproperty.(results, :normalized), player))
    n_actions = length(first(results).raw[player])
    dominant_fraction = [count(m -> m.dominant_action == action, metrics) / length(metrics) for action in 1:n_actions]
    return (;
        queries,
        solver,
        player,
        runs=length(results),
        dominant_fraction,
        positive_l2_mean=mean(getproperty.(metrics, :positive_l2)),
        positive_l2_std=std(getproperty.(metrics, :positive_l2); corrected=false),
        negative_l2_mean=mean(getproperty.(metrics, :negative_l2)),
        negative_l2_std=std(getproperty.(metrics, :negative_l2); corrected=false),
        positive_hhi_mean=finite_mean(getproperty.(metrics, :positive_hhi)),
        positive_support_mean=mean(getproperty.(metrics, :positive_support)),
    )
end

function main()
    cfg = parse_cli(ARGS)
    backup = Symbol(lowercase(cfg["backup"]))
    backup in (:sample, :mean) || error("--backup must be sample or mean")

    game = DubinMG(V=(1.0, 1.0))
    s0 = dubin_reference_state()
    oracle = ZeroSearchOracle(Tuple(length.(actions(game))))
    output = cfg["output"]
    mkpath(output)

    println("RM vs RM+ regret concentration diagnostic")
    println("state=$(s0)")
    println("runs=$(cfg["runs"]) queries=$(join(cfg["queries"], ',')) max_depth=$(cfg["max_depth"]) epsilon=$(cfg["epsilon"]) backup=$(backup)")
    println("output=$(output)")

    all_results = Dict{Tuple{Int,String},Vector{Any}}()
    for queries in cfg["queries"]
        for solver in ("rm", "rm_plus")
            rm_plus = solver == "rm_plus"
            results = Any[]
            for trial in 1:cfg["runs"]
                seed = cfg["seed"] + trial
                Random.seed!(seed)
                result = run_trial(
                    game,
                    oracle,
                    s0;
                    queries,
                    max_depth=cfg["max_depth"],
                    epsilon=cfg["epsilon"],
                    backup,
                    rm_plus,
                )
                push!(results, merge(result, (; trial, seed)))
            end
            all_results[(queries, solver)] = results

            solver_dir = joinpath(output, "queries_$(queries)", solver)
            mkpath(solver_dir)
            for player in 1:2
                write_matrix(joinpath(solver_dir, "root_raw_regret_p$(player).csv"), result_matrix(results, :raw, player))
                write_matrix(joinpath(solver_dir, "root_sqrt_normalized_regret_p$(player).csv"), result_matrix(results, :normalized, player))
                write_matrix(joinpath(solver_dir, "root_average_regret_p$(player).csv"), result_matrix(results, :average, player))
                write_matrix(joinpath(solver_dir, "root_strategy_p$(player).csv"), result_matrix(results, :strategy, player))
            end
        end
    end

    trial_header = [
        "queries", "solver", "trial", "seed", "player", "root_visits", "elapsed_seconds",
        "sqrt_normalized_positive_l1", "sqrt_normalized_positive_l2",
        "sqrt_normalized_negative_l1", "sqrt_normalized_negative_l2",
        "sqrt_normalized_signed_l2",
        "positive_hhi", "positive_entropy", "positive_support", "dominant_action",
        "sqrt_normalized_minimum_regret", "sqrt_normalized_maximum_regret",
    ]
    trial_rows = Vector{Vector{Any}}()
    for ((queries, solver), results) in sort(collect(all_results); by=first)
        for result in results, player in 1:2
            metrics = regret_metrics(result.normalized[player])
            push!(trial_rows, Any[
                queries, solver, result.trial, result.seed, player, result.visits, result.elapsed,
                metrics.positive_l1, metrics.positive_l2, metrics.negative_l1, metrics.negative_l2,
                metrics.signed_l2, metrics.positive_hhi, metrics.positive_entropy,
                metrics.positive_support, metrics.dominant_action,
                metrics.minimum_regret, metrics.maximum_regret,
            ])
        end
    end
    write_csv(joinpath(output, "trial_concentration.csv"), trial_header, trial_rows)

    n_actions = length(actions(game)[1])
    summary_header = vcat(
        ["queries", "solver", "player", "runs"],
        ["dominant_action_$(i)_fraction" for i in 1:n_actions],
        [
            "sqrt_normalized_positive_l2_mean", "sqrt_normalized_positive_l2_std",
            "sqrt_normalized_negative_l2_mean", "sqrt_normalized_negative_l2_std",
            "positive_hhi_mean", "positive_support_mean",
        ],
    )
    summary_rows = Vector{Vector{Any}}()
    for ((queries, solver), results) in sort(collect(all_results); by=first), player in 1:2
        summary = summarize_group(queries, solver, player, results)
        push!(summary_rows, Any[
            summary.queries, summary.solver, summary.player, summary.runs,
            summary.dominant_fraction...,
            summary.positive_l2_mean, summary.positive_l2_std,
            summary.negative_l2_mean, summary.negative_l2_std,
            summary.positive_hhi_mean, summary.positive_support_mean,
        ])
        @printf(
            "%5d %-7s p%d dominant=%s sqrt_pos_l2=%.4f±%.4f sqrt_neg_l2=%.4f±%.4f HHI=%.3f support=%.2f\n",
            queries,
            solver,
            player,
            string(round.(summary.dominant_fraction; digits=3)),
            summary.positive_l2_mean,
            summary.positive_l2_std,
            summary.negative_l2_mean,
            summary.negative_l2_std,
            summary.positive_hhi_mean,
            summary.positive_support_mean,
        )
    end
    write_csv(joinpath(output, "concentration_summary.csv"), summary_header, summary_rows)
    println("wrote $(output)")
end

main()
