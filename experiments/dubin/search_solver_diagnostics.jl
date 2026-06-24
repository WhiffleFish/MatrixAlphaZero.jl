using Pkg
Pkg.activate("experiments")

# Run from the repository root, for example:
#   julia --project=experiments experiments/dubin/search_solver_diagnostics.jl --solver both --runs 64 --tree_queries 1000 --max_depth 20 --br_depth 20
#
# The diagnostic uses a zero/uniform oracle so the search tree is assessed without
# any learned value, regret, or policy component.

using DelimitedFiles
using ExperimentTools
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POSGModels.Dubin
using POSGModels.StaticArrays
using Printf
using Random
using Statistics

const AZ = MatrixAlphaZero
const Tools = ExperimentTools

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
AZ.state_strategy(oracle::ZeroSearchOracle, game, s) = uniform_pair(oracle)
AZ.batch_state_strategy(oracle::ZeroSearchOracle, game, states) = (
    fill(1.0f0 / oracle.na[1], oracle.na[1], length(states)),
    fill(1.0f0 / oracle.na[2], oracle.na[2], length(states)),
)
AZ.state_regret(oracle::ZeroSearchOracle, game, s) =
    (zeros(Float32, oracle.na[1]), zeros(Float32, oracle.na[2]))
AZ.batch_state_regret(oracle::ZeroSearchOracle, game, states) =
    (zeros(Float32, oracle.na[1], length(states)), zeros(Float32, oracle.na[2], length(states)))

function parse_cli(args)
    cfg = Dict{String,Any}(
        "solver" => "both",
        "runs" => 32,
        "tree_queries" => 500,
        "max_depth" => 5,
        "br_depth" => 5,
        "epsilon" => 0.30,
        "seed" => 0,
        "backup" => "sample",
        "output" => joinpath(@__DIR__, "search_solver_diagnostics_results"),
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test"
            cfg["runs"] = 2
            cfg["tree_queries"] = 4
            cfg["max_depth"] = 2
            cfg["br_depth"] = 2
        elseif startswith(arg, "--")
            key_value = split(arg[3:end], "="; limit=2)
            key = first(key_value)
            value = if length(key_value) == 2
                last(key_value)
            else
                i < length(args) || error("Missing value for --$(key)")
                i += 1
                args[i]
            end
            if key in ("solver", "backup", "output")
                cfg[key] = value
            elseif key in ("runs", "tree_queries", "max_depth", "br_depth", "seed")
                cfg[key] = parse(Int, value)
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
    return cfg
end

function dubin_reference_state()
    return JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
end

function mcts_tree_policy_fn(game, oracle, tree, player::Int)
    return function (_, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.policy_sum[player][idx])
            return Float64.(AZ.state_policy(oracle, game, s)[player])
        end
        return AZ.normalized_or_uniform(tree.policy_sum[player][idx])
    end
end

function smoos_tree_policy_fn(game, oracle, tree, player::Int)
    return function (_, s)
        idx = findfirst(==(s), tree.s)
        if isnothing(idx) || isempty(tree.strategy[player][idx])
            return Float64.(AZ.state_strategy(oracle, game, s)[player])
        end
        return AZ.normalized_or_uniform(tree.strategy[player][idx])
    end
end

function fixed_policy_value(game, oracle, π1, π2, s, depth::Int, max_depth::Int)
    if isterminal(game, s)
        return 0.0
    elseif depth >= max_depth
        return AZ.oracle_state_value(oracle, game, s)
    end

    A1, A2 = actions(game)
    x = π1(game, s)
    y = π2(game, s)
    γ = discount(game)
    v = 0.0
    for (i, a1) in enumerate(A1), (j, a2) in enumerate(A2)
        sp, r = @gen(:sp, :r)(game, s, (a1, a2))
        v += x[i] * y[j] * (AZ.zs_reward_scalar(r) + γ * fixed_policy_value(game, oracle, π1, π2, sp, depth + 1, max_depth))
    end
    return v
end

positive_l2(x) = sqrt(sum(abs2, max.(x, 0.0)))

function entropy(p)
    s = 0.0
    for p_i in p
        if p_i > 0
            s -= p_i * log(p_i)
        end
    end
    return s
end

function summarize_tree(style::Symbol, game, oracle, search, tree, s0; br_depth::Int, epsilon::Float64, elapsed::Float64)
    if style == :regret_matching
        x, y = AZ.tree_policy(search, tree, game, 1; ϵ=epsilon)
        search_value = AZ.node_value(search, tree, game, 1, x, y)
        root_regret = (copy(tree.regret[1][1]), copy(tree.regret[2][1]))
        root_raw_regret = root_regret
        π1 = mcts_tree_policy_fn(game, oracle, tree, 1)
        π2 = mcts_tree_policy_fn(game, oracle, tree, 2)
        n_states = length(tree.s)
        n_expanded = count(!isempty, tree.s_children)
        root_visits = tree.n_s[1]
    elseif style == :smoos
        root_regret, root_strategy = AZ.root_targets(search, tree, game, 1)
        x, y = root_strategy
        search_value = fixed_policy_value(game, oracle, smoos_tree_policy_fn(game, oracle, tree, 1), smoos_tree_policy_fn(game, oracle, tree, 2), s0, 0, br_depth)
        root_raw_regret = (copy(tree.regret[1][1]), copy(tree.regret[2][1]))
        π1 = smoos_tree_policy_fn(game, oracle, tree, 1)
        π2 = smoos_tree_policy_fn(game, oracle, tree, 2)
        n_states = length(tree.s)
        n_expanded = count(!isempty, tree.regret[1])
        root_visits = search.oos_iterations
    else
        error("Unsupported style $(style)")
    end

    p1_vs_p2_br, p2_vs_p1_br = Tools.approx_br_values_both_st(
        game,
        oracle,
        π1,
        π2,
        s0;
        max_depth = br_depth,
        value_oracle = oracle,
    )
    policy_value = fixed_policy_value(game, oracle, π1, π2, s0, 0, br_depth)
    p1_br_value = -p2_vs_p1_br
    exploitability_gap = p1_br_value - p1_vs_p2_br

    return (;
        x = Float64.(x),
        y = Float64.(y),
        root_regret = (Float64.(root_regret[1]), Float64.(root_regret[2])),
        root_raw_regret = (Float64.(root_raw_regret[1]), Float64.(root_raw_regret[2])),
        search_value,
        policy_value,
        p1_vs_p2_br,
        p2_vs_p1_br,
        exploitability_gap,
        n_states,
        n_expanded,
        root_visits,
        root_regret_l2 = 0.5 * (positive_l2(root_regret[1]) + positive_l2(root_regret[2])),
        root_strategy_entropy = 0.5 * (entropy(x) + entropy(y)),
        elapsed,
    )
end

function run_mcts_trial(game, oracle, s0; tree_queries::Int, max_depth::Int, epsilon::Float64, backup::Symbol, br_depth::Int)
    search = AZ.MCTSSearch(;
        oracle,
        tree_queries,
        max_depth,
        ϵ = _ -> epsilon,
        search_style = AZ.RegretMatchingSearch(; backup),
    )
    tree = AZ.Tree(search, game, s0)
    elapsed = @elapsed begin
        for _ in 1:tree_queries
            AZ.simulate(search, tree, game, 1; ϵ=epsilon)
        end
    end
    return summarize_tree(:regret_matching, game, oracle, search, tree, s0; br_depth, epsilon, elapsed)
end

function run_smoos_trial(game, oracle, s0; oos_iterations::Int, max_depth::Int, epsilon::Float64, br_depth::Int)
    search = AZ.SMOOSSearch(;
        oracle,
        oos_iterations,
        max_depth,
        ϵ = _ -> epsilon,
        τ = 0.0,
        transfer_weight = 0.0,
    )
    tree = AZ.Tree(search, game, s0)
    elapsed = @elapsed begin
        for _ in 1:oos_iterations
            AZ.smoos_trajectory!(search, tree, game, 1, 0, 1.0, 1.0, 1.0, 1.0; ϵ=epsilon)
        end
    end
    return summarize_tree(:smoos, game, oracle, search, tree, s0; br_depth, epsilon, elapsed)
end

function matrix_from(results, property::Symbol, player::Int)
    return reduce(hcat, map(r -> getproperty(r, property)[player], results))'
end

function write_matrix(path::String, mat)
    header = reshape(["action_$(i)" for i in axes(mat, 2)], 1, :)
    writedlm(path, [header; mat], ',')
end

function write_results(output_dir::String, style_name::String, results, cfg)
    style_dir = joinpath(output_dir, style_name)
    mkpath(style_dir)

    summary_header = [
        "style" "trial" "seed" "iterations" "max_depth" "br_depth" "epsilon" "backup" "n_states" "n_expanded" "root_visits" "search_value" "policy_value" "p1_value_vs_p2_br" "p2_value_vs_p1_br" "exploitability_gap" "root_regret_l2" "root_strategy_entropy" "elapsed_seconds"
    ]
    rows = mapreduce(vcat, enumerate(results)) do (i, r)
        [
            style_name i r.seed cfg["tree_queries"] cfg["max_depth"] cfg["br_depth"] cfg["epsilon"] cfg["backup"] r.n_states r.n_expanded r.root_visits r.search_value r.policy_value r.p1_vs_p2_br r.p2_vs_p1_br r.exploitability_gap r.root_regret_l2 r.root_strategy_entropy r.elapsed
        ]
    end
    writedlm(joinpath(style_dir, "summary.csv"), [summary_header; rows], ',')
    write_matrix(joinpath(style_dir, "root_regret_p1.csv"), matrix_from(results, :root_regret, 1))
    write_matrix(joinpath(style_dir, "root_regret_p2.csv"), matrix_from(results, :root_regret, 2))
    write_matrix(joinpath(style_dir, "root_raw_regret_p1.csv"), matrix_from(results, :root_raw_regret, 1))
    write_matrix(joinpath(style_dir, "root_raw_regret_p2.csv"), matrix_from(results, :root_raw_regret, 2))
    write_matrix(joinpath(style_dir, "root_strategy_p1.csv"), reduce(hcat, getproperty.(results, :x))')
    write_matrix(joinpath(style_dir, "root_strategy_p2.csv"), reduce(hcat, getproperty.(results, :y))')
    return style_dir
end

function print_summary(style_name::String, results)
    exploitability = getproperty.(results, :exploitability_gap)
    values = getproperty.(results, :policy_value)
    regrets = getproperty.(results, :root_regret_l2)
    entropy_vals = getproperty.(results, :root_strategy_entropy)
    @printf(
        "%s: exploitability_gap mean=%.5f std=%.5f min=%.5f max=%.5f | value mean=%.5f | regret_l2 mean=%.5f | entropy mean=%.5f\n",
        style_name,
        mean(exploitability),
        length(exploitability) > 1 ? std(exploitability) : 0.0,
        minimum(exploitability),
        maximum(exploitability),
        mean(values),
        mean(regrets),
        mean(entropy_vals),
    )
end

function main()
    cfg = parse_cli(ARGS)
    solver = lowercase(cfg["solver"])
    solver in ("both", "regret_matching", "smoos") || error("--solver must be one of both, regret_matching, smoos")
    backup = Symbol(lowercase(cfg["backup"]))
    backup in (:sample, :mean) || error("--backup must be sample or mean")

    game = DubinMG(V = (1.0, 1.0))
    s0 = dubin_reference_state()
    na = Tuple(length.(actions(game)))
    oracle = ZeroSearchOracle(na)
    styles = solver == "both" ? ("regret_matching", "smoos") : (solver,)
    mkpath(cfg["output"])

    println("Dubin search solver diagnostics")
    println("state=$(s0)")
    println("runs=$(cfg["runs"]) iterations=$(cfg["tree_queries"]) max_depth=$(cfg["max_depth"]) br_depth=$(cfg["br_depth"]) epsilon=$(cfg["epsilon"])")
    println("output=$(cfg["output"])")

    for (style_idx, style_name) in enumerate(styles)
        results = map(1:cfg["runs"]) do trial
            seed = cfg["seed"] + 10_000 * style_idx + trial
            Random.seed!(seed)
            result = if style_name == "regret_matching"
                run_mcts_trial(
                    game,
                    oracle,
                    s0;
                    tree_queries = cfg["tree_queries"],
                    max_depth = cfg["max_depth"],
                    epsilon = cfg["epsilon"],
                    backup,
                    br_depth = cfg["br_depth"],
                )
            else
                run_smoos_trial(
                    game,
                    oracle,
                    s0;
                    oos_iterations = cfg["tree_queries"],
                    max_depth = cfg["max_depth"],
                    epsilon = cfg["epsilon"],
                    br_depth = cfg["br_depth"],
                )
            end
            return merge(result, (; seed))
        end
        style_dir = write_results(cfg["output"], style_name, results, cfg)
        print_summary(style_name, results)
        println("  wrote $(style_dir)")
    end
end

main()
