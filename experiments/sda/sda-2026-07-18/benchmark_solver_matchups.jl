using Serialization

include("ppo_solver_response_utilities.jl")

const BENCHMARK_COLUMNS = ("uniform_random", "heuristic", "ppo_br")
const PPO_RESULTS_DIR = joinpath(EXPERIMENT_DIR, "ppo_solver_response_utility_results")
const BENCHMARK_CONTEXT_CACHE = Dict{String,Any}()

function parse_benchmark_args(args)
    cfg = Dict{String,String}(
        "runs" => "500",
        "max-steps" => "50",
        "seed" => "20260720",
        "iter" => "latest",
        "output-dir" => joinpath(EXPERIMENT_DIR, "solver_benchmark_results"),
        "workers" => string(min(6, max(Sys.CPU_THREADS - 1, 1))),
        "worker-jobs" => "",
        "worker-output" => "",
        "test" => "false",
    )
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        opt = key[3:end]
        haskey(cfg, opt) || error("Unknown option $(key)")
        if opt == "test"
            cfg[opt] = "true"
            i += 1
            continue
        end
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        cfg[opt] = args[i]
        i += 1
    end
    parsed = (;
        runs=parse(Int, cfg["runs"]),
        max_steps=parse(Int, cfg["max-steps"]),
        seed=parse(Int, cfg["seed"]),
        iter=cfg["iter"],
        output_dir=abspath(cfg["output-dir"]),
        workers=parse(Int, cfg["workers"]),
        worker_jobs=cfg["worker-jobs"],
        worker_output=cfg["worker-output"],
        test=parse(Bool, cfg["test"]),
    )
    parsed.runs > 0 || error("--runs must be positive")
    parsed.max_steps > 0 || error("--max-steps must be positive")
    parsed.workers >= 0 || error("--workers must be nonnegative")
    return parsed.test ? merge(parsed, (; runs=2)) : parsed
end

function uniform_player_policy(game, player::Int)
    n = length(actions(game)[player])
    probs = fill(inv(n), n)
    return Tools.FunctionPlayerPolicy(game, player, (_game, _state) -> probs)
end

heuristic_player_policy(game, player::Int) =
    Tools.sda_no_burn_heuristic(game, player)

function ppo_model_path(solver::AbstractString, player::Int)
    return joinpath(
        PPO_RESULTS_DIR,
        solver,
        "p$(player)",
        "ppo_response_actor_critic.jld2",
    )
end

function ppo_player_policy(game, solver::AbstractString, player::Int)
    path = ppo_model_path(solver, player)
    isfile(path) || error("Missing PPO response model: $(path)")
    data = JLD2.load(path)
    actor = data["actor"]
    metadata = data["metadata"]
    get(metadata, "solver", nothing) == solver ||
        error("PPO model solver mismatch in $(path)")
    get(metadata, "response_player", nothing) == player ||
        error("PPO model player mismatch in $(path)")
    return Tools.ActorPlayerPolicy(game, player, actor)
end

function opponent_policy(
        game,
        solver::AbstractString,
        opponent::Int,
        benchmark::AbstractString,
    )
    benchmark == "uniform_random" && return uniform_player_policy(game, opponent)
    benchmark == "heuristic" && return heuristic_player_policy(game, opponent)
    benchmark == "ppo_br" && return ppo_player_policy(game, solver, opponent)
    error("Unknown benchmark $(benchmark)")
end

function joint_matchup(tree_policy, opponent_policy, tree_player::Int)
    return isone(tree_player) ?
        Tools.JointPolicy(tree_policy, opponent_policy) :
        Tools.JointPolicy(opponent_policy, tree_policy)
end

function evaluate_matchup(
        game,
        joint_policy,
        tree_player::Int,
        initialstates,
        cfg,
    )
    # Use the same sampling stream for every cell in a player table.
    Random.seed!(cfg.seed + 1_000 * tree_player)
    result = Tools.evaluate_joint_policy(
        game,
        joint_policy,
        cfg.runs;
        max_steps=cfg.max_steps,
        initialstates,
        show_progress=false,
        proc_warn=false,
        parallel=false,
    )
    return result.reward[tree_player], result.stderr_reward[tree_player]
end

function write_benchmark_table(path, values)
    header = reshape(["solver"; collect(BENCHMARK_COLUMNS)], 1, :)
    data = reduce(vcat, [
        reshape(Any[solver; values[solver]...], 1, :)
        for solver in SOLVERS
    ])
    mkpath(dirname(path))
    writedlm(path, [header; data], ',')
    return path
end

function benchmark_context(cfg)
    key = "$(cfg.iter)|$(cfg.test)"
    return get!(BENCHMARK_CONTEXT_CACHE, key) do
        game = make_game()
        oracle, model_iter, checkpoint = load_checkpoint_oracle(cfg.iter)
        solver_cfg = parse_args(String[])
        solver_cfg = merge(
            solver_cfg,
            (;
                iter=cfg.iter,
                source_mass=Float64(model_iter * solver_cfg.train_tree_queries),
            ),
        )
        cfg.test && (solver_cfg = merge(solver_cfg, (; tree_queries=2, max_depth=2)))
        return (; game, oracle, solver_cfg, checkpoint)
    end
end

function run_benchmark_job(job, initialstates, cfg)
    context = benchmark_context(cfg)
    tree_player, solver, benchmark = job
    opponent = MarkovGames.other_player(tree_player)
    planner, _ = solver_policy(solver, context.game, context.oracle, context.solver_cfg)
    tree_policy = Tools.SinglePlayerAlphaZeroPolicy(planner, tree_player)
    fixed = opponent_policy(context.game, solver, opponent, benchmark)
    joint_policy = joint_matchup(tree_policy, fixed, tree_player)
    utility, stderr = evaluate_matchup(
        context.game,
        joint_policy,
        tree_player,
        initialstates,
        cfg,
    )
    println(
        "p$(tree_player) solver=$(solver) opponent=$(benchmark) ",
        "utility=$(round(utility; digits=5)) ",
        "stderr=$(round(stderr; digits=5))",
    )
    return (; tree_player, solver, benchmark, utility, stderr)
end

function run_benchmark_jobs(jobs, initialstates, cfg)
    cfg.workers == 0 && return map(job -> run_benchmark_job(job, initialstates, cfg), jobs)
    worker_count = min(cfg.workers, length(jobs))
    groups = [collect(worker:worker_count:length(jobs)) for worker in 1:worker_count]
    return mktempdir() do temp_dir
        processes = Base.Process[]
        result_paths = String[]
        for (worker, indices) in enumerate(groups)
            result_path = joinpath(temp_dir, "worker$(worker).bin")
            args = [
                Base.julia_cmd().exec;
                "--project=$(dirname(Base.active_project()))";
                abspath(@__FILE__);
                "--runs"; string(cfg.runs);
                "--max-steps"; string(cfg.max_steps);
                "--seed"; string(cfg.seed);
                "--iter"; cfg.iter;
                "--workers"; "0";
                "--worker-jobs"; join(indices, ",");
                "--worker-output"; result_path;
            ]
            cfg.test && push!(args, "--test")
            push!(processes, run(Cmd(args); wait=false))
            push!(result_paths, result_path)
        end
        foreach(wait, processes)
        failed = findall(process -> !success(process), processes)
        isempty(failed) || error("Benchmark subprocesses failed: $(join(failed, ", "))")
        return reduce(vcat, [open(deserialize, path) for path in result_paths])
    end
end

function collect_player_tables(results, tree_player::Int)
    utilities = Dict(solver => zeros(length(BENCHMARK_COLUMNS)) for solver in SOLVERS)
    stderrs = Dict(solver => zeros(length(BENCHMARK_COLUMNS)) for solver in SOLVERS)
    for result in results
        result.tree_player == tree_player || continue
        column = findfirst(==(result.benchmark), BENCHMARK_COLUMNS)
        utilities[result.solver][column] = result.utility
        stderrs[result.solver][column] = result.stderr
    end
    return utilities, stderrs
end

function main_benchmark(args)
    cfg = parse_benchmark_args(args)
    game = make_game()
    initialstate_rng = MersenneTwister(cfg.seed + 99)
    initialstates = [rand(initialstate_rng, initialstate(game)) for _ in 1:cfg.runs]
    jobs = [
        (tree_player, solver, benchmark)
        for tree_player in (1, 2)
        for solver in SOLVERS
        for benchmark in BENCHMARK_COLUMNS
    ]
    if !isempty(cfg.worker_jobs)
        isempty(cfg.worker_output) && error("--worker-output is required with --worker-jobs")
        indices = parse.(Int, split(cfg.worker_jobs, ','))
        results = map(index -> run_benchmark_job(jobs[index], initialstates, cfg), indices)
        open(cfg.worker_output, "w") do io
            serialize(io, results)
        end
        return results
    end

    mkpath(cfg.output_dir)
    println("Evaluation state: shared SDA game-distribution state bank")
    println("Evaluating 18 matchups with $(cfg.workers) worker processes")
    results = run_benchmark_jobs(jobs, initialstates, cfg)

    paths = String[]
    for tree_player in (1, 2)
        utilities, stderrs = collect_player_tables(results, tree_player)
        push!(paths, write_benchmark_table(
            joinpath(cfg.output_dir, "player$(tree_player)_utilities.csv"),
            utilities,
        ))
        push!(paths, write_benchmark_table(
            joinpath(cfg.output_dir, "player$(tree_player)_stderrs.csv"),
            stderrs,
        ))
    end
    foreach(path -> println("Wrote benchmark table: $(path)"), paths)
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_benchmark(ARGS)
end
