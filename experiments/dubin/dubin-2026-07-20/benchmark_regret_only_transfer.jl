using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using POSGModels.Dubin
using Random

include(joinpath(@__DIR__, "ppo_solver_response_utilities.jl"))

const DEFAULT_REGRET_ONLY_OUTPUT_DIR = joinpath(
    @__DIR__,
    "regret_only_transfer_screen",
)

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function transfer_candidate(
        name;
        scale=0.0,
        backup=:mean,
        regret_weight=0.0,
        strategy_weight=0.0,
        statistic_weight=0.0,
        reach_power=1.0,
    )
    return (;
        name,
        scale=Float64(scale),
        backup=Symbol(backup),
        regret_weight=Float64(regret_weight),
        strategy_weight=Float64(strategy_weight),
        statistic_weight=Float64(statistic_weight),
        reach_power=Float64(reach_power),
    )
end

function screening_candidates()
    candidates = [
        transfer_candidate("value_only_mean"),
        transfer_candidate(
            "regret_only_sample_s5";
            scale=5.0,
            backup=:sample,
            regret_weight=1.0,
        ),
        transfer_candidate(
            "coupled_mean_s5";
            scale=5.0,
            regret_weight=1.0,
            strategy_weight=1.0,
            statistic_weight=1.0,
        ),
    ]
    for scale in (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)
        push!(candidates, transfer_candidate(
            "regret_only_mean_s$(scale)";
            scale,
            regret_weight=1.0,
        ))
    end
    return candidates
end

function selected_candidates(candidates, names)
    isempty(names) && return candidates
    requested = Set(split(names, ','))
    selected = filter(candidate -> candidate.name in requested, candidates)
    found = Set(candidate.name for candidate in selected)
    missing = setdiff(requested, found)
    isempty(missing) ||
        error("Unknown candidates: $(join(sort!(collect(missing)), ", "))")
    return selected
end

function build_transfer_search(
        oracle,
        candidate;
        queries,
        max_depth,
        search_epsilon,
    )
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=queries,
        max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(;
            backup=candidate.backup,
            method=AZ.Plus(),
        ),
        value_target=:search,
        ϵ=_ -> search_epsilon,
        prior_scale=candidate.scale,
        regret_prior_weight=candidate.regret_weight,
        strategy_prior_weight=candidate.strategy_weight,
        statistic_prior_weight=candidate.statistic_weight,
        prior_reach_power=candidate.reach_power,
    )
end

function search_oracle(game, learned_oracle, candidate)
    candidate.name == "value_only_mean" &&
        return ValueOnlySearchOracle(game, learned_oracle)
    return learned_oracle
end

function heuristic_player_policy(game, player)
    return player == 1 ?
        DubinTools.dubin_attacker_heuristic(game) :
        DubinTools.dubin_defender_heuristic(game)
end

function evaluate_against_heuristic(
        game,
        learned_oracle,
        candidate,
        search_player,
        initialstates;
        max_steps,
        queries,
        max_depth,
        search_epsilon,
        seed,
    )
    oracle = search_oracle(game, learned_oracle, candidate)
    search = build_transfer_search(
        oracle,
        candidate;
        queries,
        max_depth,
        search_epsilon,
    )
    planner = AZ.AlphaZeroPlanner(game, search)
    search_policy = Tools.SinglePlayerAlphaZeroPolicy(planner, search_player)
    opponent = heuristic_player_policy(game, 3 - search_player)
    joint_policy = search_player == 1 ?
        Tools.JointPolicy(search_policy, opponent) :
        Tools.JointPolicy(opponent, search_policy)

    println(
        "[dubin-transfer] start candidate=$(candidate.name) ",
        "player=$(search_player) runs=$(length(initialstates))",
    )
    flush(stdout)
    Random.seed!(seed + search_player)
    elapsed = @elapsed result = rollout_eval(
        game,
        joint_policy,
        initialstates;
        runs=length(initialstates),
        max_steps,
    )
    reward = result.reward[search_player]
    stderr = result.stderr_reward[search_player]
    println(
        "[dubin-transfer] done candidate=$(candidate.name) ",
        "player=$(search_player) reward=$(round(reward; digits=6)) ",
        "stderr=$(round(stderr; digits=6)) elapsed=$(round(elapsed; digits=1))s",
    )
    flush(stdout)
    return merge(candidate, (;
        search_player,
        runs=length(initialstates),
        reward,
        stderr_reward=stderr,
        mean_steps=result.mean_steps,
        attacker_goal_rate=result.attacker_goal_rate,
        tagged_rate=result.tagged_rate,
        timeout_rate=result.timeout_rate,
        queries,
        max_depth,
        max_steps,
        search_epsilon,
        action_epsilon=0.0,
        elapsed_seconds=elapsed,
    ))
end

function csv_value(value)
    value isa AbstractString && return value
    value isa Symbol && return String(value)
    value isa Real && return isfinite(value) ? string(value) : ""
    return string(value)
end

function write_csv(path, rows)
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
    return path
end

function main(args=ARGS)
    test = "--test" in args
    runs = option_value(args, "--runs", 200, x -> parse(Int, x))
    queries = option_value(args, "--tree-queries", 100, x -> parse(Int, x))
    max_depth = option_value(args, "--max-depth", 5, x -> parse(Int, x))
    max_steps = option_value(args, "--max-steps", 50, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    seed = option_value(args, "--seed", 20260724, x -> parse(Int, x))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    only_names = option_value(args, "--only", "", String)
    output_dir = abspath(option_value(
        args,
        "--output-dir",
        DEFAULT_REGRET_ONLY_OUTPUT_DIR,
        String,
    ))
    if test
        runs = min(runs, 2)
        queries = min(queries, 2)
        max_depth = min(max_depth, 2)
        max_steps = min(max_steps, 3)
        only_names = "value_only_mean,regret_only_mean_s5.0"
    end

    runs > 0 || error("--runs must be positive")
    queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    0 < max_steps <= MAX_PPO_STEPS ||
        error("--max-steps must be in 1:$(MAX_PPO_STEPS)")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")

    game = DubinMG(V=(1.0, 1.0))
    learned_oracle, model_iter, checkpoint = load_checkpoint_oracle(checkpoint_spec)
    initialstates = fill(initial_dubin_state(), runs)
    candidates = selected_candidates(screening_candidates(), only_names)

    println("[dubin-transfer] checkpoint=$(checkpoint) iteration=$(model_iter)")
    println(
        "[dubin-transfer] runs=$(runs) queries=$(queries) depth=$(max_depth) ",
        "steps=$(max_steps) search_epsilon=$(search_epsilon) action_epsilon=0.0",
    )
    flush(stdout)

    rows = NamedTuple[]
    for candidate in candidates, search_player in 1:2
        push!(rows, evaluate_against_heuristic(
            game,
            learned_oracle,
            candidate,
            search_player,
            initialstates;
            max_steps,
            queries,
            max_depth,
            search_epsilon,
            seed,
        ))
    end
    baseline = Dict(
        row.search_player => row.reward
        for row in rows
        if row.name == "value_only_mean"
    )
    if length(baseline) == 2
        rows = map(rows) do row
            merge(row, (delta_vs_value_only=row.reward - baseline[row.search_player],))
        end
    end

    mkpath(output_dir)
    output = write_csv(joinpath(output_dir, "heuristic_matchups.csv"), rows)
    println("[dubin-transfer] wrote $(output)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
