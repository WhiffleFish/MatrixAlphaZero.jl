using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPs
using ProgressMeter
using Random
using Statistics

const AZ = MatrixAlphaZero

function residual_option(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function residual_flag(args, name)
    return name in args
end

function checkpoint_iteration(path::AbstractString)
    matched = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(matched) && error("Invalid checkpoint name: $(path)")
    return parse(Int, only(matched.captures))
end

function checkpoint_paths(models_dir)
    isdir(models_dir) || error("Missing checkpoint directory: $(models_dir)")
    paths = filter(
        path -> occursin(r"oracle\d+\.jld2$", basename(path)),
        readdir(models_dir; join=true),
    )
    isempty(paths) && error("No checkpoints found in $(models_dir)")
    sort!(paths; by=checkpoint_iteration)
    return paths
end

function select_checkpoints(paths; every, first_iteration, last_iteration)
    selected = filter(paths) do path
        iteration = checkpoint_iteration(path)
        first_iteration <= iteration <= last_iteration &&
            iszero(mod(iteration, every))
    end
    eligible = filter(
        path -> first_iteration <= checkpoint_iteration(path) <= last_iteration,
        paths,
    )
    isempty(eligible) && error(
        "No checkpoints in requested range $(first_iteration):$(last_iteration)",
    )
    final_path = last(eligible)
    final_path in selected || push!(selected, final_path)
    sort!(unique!(selected); by=checkpoint_iteration)
    return selected
end

function resolve_checkpoint(paths, spec::AbstractString)
    spec == "latest" && return last(paths)
    spec == "first" && return first(paths)
    iteration = parse(Int, spec)
    matches = filter(path -> checkpoint_iteration(path) == iteration, paths)
    isempty(matches) && error("No checkpoint for iteration $(iteration)")
    return only(matches)
end

function residual_search(config, oracle, iteration, args)
    # Checkpoint k is the oracle available at the start of outer iteration
    # k + 1, so this matches the exploration schedule used by the next
    # self-play/data-gathering step.
    epsilon = config.epsilon_schedule(iteration + 1)
    return AZ.MCTSSearch(;
        oracle,
        tree_queries=args.tree_queries,
        max_depth=args.max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(;
            backup=config.backup,
            method=AZ.Plus(),
        ),
        value_target=:search,
        ϵ=_ -> epsilon,
        prior_scale=0.0,
    ), epsilon
end

function oracle_value(oracle, state_features)
    return Float64(only(AZ.value(oracle, state_features)))
end

function search_backup(search, game, state, epsilon)
    search_start = time()
    (_, _, backup), _ = AZ.search_info(search, game, state; ϵ=epsilon)
    return Float64(backup), time() - search_start
end

function selfplay_episode(
        search,
        game,
        initial_state;
        search_epsilon,
        action_epsilon,
        sim_depth,
    )
    A1, A2 = actions(game)
    states = Any[]
    features = Vector{Float32}[]
    value_predictions = Float64[]
    backups = Float64[]
    search_times = Float64[]
    state = initial_state
    step = 1
    while step <= sim_depth && !isterminal(game, state)
        push!(states, state)
        state_features = MarkovGames.convert_s(Vector{Float32}, state, game)
        push!(features, state_features)
        push!(value_predictions, oracle_value(search.oracle, state_features))

        search_start = time()
        (_, _, backup), info =
            AZ.search_info(search, game, state; ϵ=search_epsilon)
        _, strategies = AZ.mcts_root_targets(search, info.tree, game, 1)
        push!(backups, Float64(backup))
        push!(search_times, time() - search_start)

        x = AZ.eps_exploration(
            AZ.normalized_or_uniform(strategies[1]),
            action_epsilon,
        )
        y = AZ.eps_exploration(
            AZ.normalized_or_uniform(strategies[2]),
            action_epsilon,
        )
        action_indices = Tuple(AZ.action_idx_from_probs(x, y))
        action = (A1[action_indices[1]], A2[action_indices[2]])
        state = @gen(:sp)(game, state, action)
        step += 1
    end
    return (; states, features, value_predictions, backups, search_times)
end

function load_checkpoint!(oracle, path)
    Flux.loadmodel!(oracle, path)
    return oracle
end

function collect_fixed_state_bank(
        config,
        oracle,
        checkpoint,
        args,
    )
    iteration = checkpoint_iteration(checkpoint)
    load_checkpoint!(oracle, checkpoint)
    search, epsilon = residual_search(config, oracle, iteration, args)
    Random.seed!(args.seed + 91_337)

    states = Any[]
    features = Vector{Float32}[]
    source_episodes = Int[]
    source_steps = Int[]
    episode = 0
    empty_episodes = 0
    progress = Progress(
        args.samples_per_checkpoint;
        desc="collect fixed state bank",
    )
    while length(states) < args.samples_per_checkpoint
        episode += 1
        history = selfplay_episode(
            search,
            config.game,
            rand(config.initialstate_distribution);
            search_epsilon=epsilon,
            action_epsilon=epsilon,
            sim_depth=args.sim_depth,
        )
        if isempty(history.states)
            empty_episodes += 1
            empty_episodes <= 100 ||
                error("Unable to collect nonterminal state-bank samples")
            continue
        end
        for step in eachindex(history.states)
            length(states) >= args.samples_per_checkpoint && break
            push!(states, history.states[step])
            push!(features, history.features[step])
            push!(source_episodes, episode)
            push!(source_steps, step)
            next!(progress)
        end
    end
    finish!(progress)
    return (;
        states,
        features,
        source_episodes,
        source_steps,
        source_checkpoint=fill(iteration, length(states)),
    )
end

function fixed_bank_samples(
        config,
        oracle,
        checkpoint,
        bank,
        args,
    )
    iteration = checkpoint_iteration(checkpoint)
    load_checkpoint!(oracle, checkpoint)
    search, epsilon = residual_search(config, oracle, iteration, args)
    Random.seed!(args.seed)
    n = length(bank.states)
    value_predictions = Vector{Float64}(undef, n)
    backups = Vector{Float64}(undef, n)
    search_times = Vector{Float64}(undef, n)
    progress = Progress(n; desc="checkpoint $(iteration)")
    for index in eachindex(bank.states)
        value_predictions[index] =
            oracle_value(oracle, bank.features[index])
        backups[index], search_times[index] = search_backup(
            search,
            config.game,
            bank.states[index],
            epsilon,
        )
        next!(progress)
    end
    finish!(progress)
    return (;
        features=bank.features,
        episodes=bank.source_episodes,
        steps=bank.source_steps,
        source_checkpoints=bank.source_checkpoint,
        value_predictions,
        backups,
        search_times,
        epsilon,
    )
end

function on_policy_samples(
        config,
        oracle,
        checkpoint,
        args,
    )
    iteration = checkpoint_iteration(checkpoint)
    load_checkpoint!(oracle, checkpoint)
    search, epsilon = residual_search(config, oracle, iteration, args)
    Random.seed!(args.seed)

    features = Vector{Float32}[]
    episodes = Int[]
    steps = Int[]
    value_predictions = Float64[]
    backups = Float64[]
    search_times = Float64[]
    episode = 0
    empty_episodes = 0
    progress = Progress(
        args.samples_per_checkpoint;
        desc="checkpoint $(iteration)",
    )
    while length(features) < args.samples_per_checkpoint
        episode += 1
        history = selfplay_episode(
            search,
            config.game,
            rand(config.initialstate_distribution);
            search_epsilon=epsilon,
            action_epsilon=epsilon,
            sim_depth=args.sim_depth,
        )
        if isempty(history.features)
            empty_episodes += 1
            empty_episodes <= 100 ||
                error("Unable to collect nonterminal on-policy samples")
            continue
        end
        for step in eachindex(history.features)
            length(features) >= args.samples_per_checkpoint && break
            push!(features, history.features[step])
            push!(episodes, episode)
            push!(steps, step)
            push!(value_predictions, history.value_predictions[step])
            push!(backups, history.backups[step])
            push!(search_times, history.search_times[step])
            next!(progress)
        end
    end
    finish!(progress)
    return (;
        features,
        episodes,
        steps,
        source_checkpoints=fill(iteration, length(features)),
        value_predictions,
        backups,
        search_times,
        epsilon,
    )
end

function residual_statistics(
        iteration,
        checkpoint,
        samples,
        args,
        state_mode,
    )
    residuals = samples.backups .- samples.value_predictions
    absolute = abs.(residuals)
    root_indices = findall(==(1), samples.steps)
    root_residuals = residuals[root_indices]
    root_absolute = absolute[root_indices]
    root_l2 = isempty(root_residuals) ?
        NaN : sqrt(mean(abs2, root_residuals))
    root_linf = isempty(root_absolute) ? NaN : maximum(root_absolute)
    return (;
        iteration,
        checkpoint,
        state_mode,
        samples=length(residuals),
        root_samples=length(root_indices),
        tree_queries=args.tree_queries,
        max_depth=args.max_depth,
        sim_depth=args.sim_depth,
        search_epsilon=samples.epsilon,
        action_epsilon=samples.epsilon,
        backup=String(args.backup),
        value_target="search",
        prior_scale=0.0,
        mean_value_prediction=mean(samples.value_predictions),
        mean_search_backup=mean(samples.backups),
        residual_mean=mean(residuals),
        residual_stderr=std(residuals) / sqrt(length(residuals)),
        residual_l1=mean(absolute),
        residual_l2=sqrt(mean(abs2, residuals)),
        residual_linf=maximum(absolute),
        residual_abs_median=quantile(absolute, 0.5),
        residual_abs_p90=quantile(absolute, 0.9),
        residual_abs_p95=quantile(absolute, 0.95),
        root_residual_mean=isempty(root_residuals) ?
            NaN : mean(root_residuals),
        root_residual_l1=isempty(root_absolute) ?
            NaN : mean(root_absolute),
        root_residual_l2=root_l2,
        root_residual_linf=root_linf,
        mean_search_seconds=mean(samples.search_times),
        total_search_seconds=sum(samples.search_times),
    )
end

csv_value(value::AbstractString) =
    occursin(',', value) ? "\"$(replace(value, "\"" => "\"\""))\"" : value
csv_value(value) = string(value)

function write_csv_header(io, columns)
    println(io, join(String.(columns), ','))
    flush(io)
end

function write_csv_row(io, row, columns)
    println(io, join(
        (csv_value(getproperty(row, column)) for column in columns),
        ',',
    ))
    flush(io)
end

function write_sample_rows(
        io,
        iteration,
        samples,
        state_dim,
    )
    for index in eachindex(samples.features)
        residual = samples.backups[index] - samples.value_predictions[index]
        values = Any[
            iteration,
            index,
            samples.source_checkpoints[index],
            samples.episodes[index],
            samples.steps[index],
            samples.steps[index] == 1,
            samples.value_predictions[index],
            samples.backups[index],
            residual,
            abs(residual),
            residual^2,
            samples.search_times[index],
        ]
        append!(values, samples.features[index])
        println(io, join(csv_value.(values), ','))
    end
    flush(io)
end

function write_metadata(path, config, args, checkpoints, bank_checkpoint)
    rows = [
        ("experiment", config.name),
        ("state_mode", args.state_mode),
        ("checkpoint_every", string(args.every)),
        ("checkpoint_first", string(checkpoint_iteration(first(checkpoints)))),
        ("checkpoint_last", string(checkpoint_iteration(last(checkpoints)))),
        ("checkpoint_count", string(length(checkpoints))),
        ("bank_checkpoint", isnothing(bank_checkpoint) ? "" :
            string(checkpoint_iteration(bank_checkpoint))),
        ("samples_per_checkpoint", string(args.samples_per_checkpoint)),
        ("tree_queries", string(args.tree_queries)),
        ("max_depth", string(args.max_depth)),
        ("sim_depth", string(args.sim_depth)),
        ("backup", String(args.backup)),
        ("value_target", "search"),
        ("prior_scale", "0.0"),
        ("seed", string(args.seed)),
        ("workers", string(args.workers)),
        ("initial_distribution", config.initial_distribution_name),
        (
            "residual_definition",
            "search_backup_minus_current_value_prediction",
        ),
    ]
    open(path, "w") do io
        println(io, "key,value")
        for (key, value) in rows
            println(io, "$(csv_value(key)),$(csv_value(value))")
        end
    end
    return path
end

function parse_residual_args(config, raw_args)
    test = residual_flag(raw_args, "--test")
    worker = residual_flag(raw_args, "--worker")
    state_mode = replace(
        residual_option(raw_args, "--state-mode", "fixed", String),
        '-' => '_',
    )
    state_mode in ("fixed", "on_policy") ||
        error("--state-mode must be fixed or on_policy")
    every = residual_option(raw_args, "--every", 10, x -> parse(Int, x))
    samples_per_checkpoint = residual_option(
        raw_args,
        "--samples-per-checkpoint",
        256,
        x -> parse(Int, x),
    )
    tree_queries = residual_option(
        raw_args,
        "--tree-queries",
        config.tree_queries,
        x -> parse(Int, x),
    )
    max_depth = residual_option(
        raw_args,
        "--max-depth",
        config.max_depth,
        x -> parse(Int, x),
    )
    sim_depth = residual_option(
        raw_args,
        "--sim-depth",
        config.sim_depth,
        x -> parse(Int, x),
    )
    first_iteration = residual_option(
        raw_args,
        "--first",
        0,
        x -> parse(Int, x),
    )
    last_iteration_spec =
        residual_option(raw_args, "--last", "latest", String)
    all_paths = checkpoint_paths(config.models_dir)
    last_iteration = last_iteration_spec == "latest" ?
        checkpoint_iteration(last(all_paths)) : parse(Int, last_iteration_spec)
    bank_checkpoint_spec =
        residual_option(raw_args, "--bank-checkpoint", "latest", String)
    seed = residual_option(raw_args, "--seed", 20260727, x -> parse(Int, x))
    workers = residual_option(
        raw_args,
        "--workers",
        min(4, max(Sys.CPU_THREADS - 1, 1)),
        x -> parse(Int, x),
    )
    checkpoint_iterations = residual_option(
        raw_args,
        "--checkpoint-iterations",
        "",
        String,
    )
    state_bank_path = residual_option(
        raw_args,
        "--state-bank-path",
        "",
        String,
    )
    output_dir = abspath(residual_option(
        raw_args,
        "--output-dir",
        joinpath(config.output_dir, state_mode),
        String,
    ))
    if test
        samples_per_checkpoint = min(samples_per_checkpoint, 4)
        tree_queries = min(tree_queries, 2)
        max_depth = min(max_depth, 2)
        sim_depth = min(sim_depth, 3)
    end

    every > 0 || error("--every must be positive")
    samples_per_checkpoint > 0 ||
        error("--samples-per-checkpoint must be positive")
    tree_queries > 0 || error("--tree-queries must be positive")
    max_depth > 0 || error("--max-depth must be positive")
    sim_depth > 0 || error("--sim-depth must be positive")
    first_iteration >= 0 || error("--first must be nonnegative")
    last_iteration >= first_iteration ||
        error("--last must be at least --first")
    workers >= 0 || error("--workers must be nonnegative")
    return (;
        test,
        worker,
        state_mode,
        every,
        samples_per_checkpoint,
        tree_queries,
        max_depth,
        sim_depth,
        first_iteration,
        last_iteration,
        bank_checkpoint_spec,
        seed,
        workers,
        checkpoint_iterations,
        state_bank_path,
        output_dir,
        backup=config.backup,
        all_paths,
    )
end

function explicit_checkpoints(paths, specification)
    isempty(specification) && return nothing
    iterations = parse.(Int, split(specification, ','))
    return map(iterations) do iteration
        matches = filter(
            path -> checkpoint_iteration(path) == iteration,
            paths,
        )
        isempty(matches) && error("No checkpoint for iteration $(iteration)")
        only(matches)
    end
end

function load_fixed_state_bank(path)
    isfile(path) || error("Missing fixed state bank: $(path)")
    data = JLD2.load(path)
    states = data["states"]
    features = data["features"]
    source_episodes = Int.(data["source_episodes"])
    source_steps = Int.(data["source_steps"])
    source_checkpoint = Int.(data["source_checkpoint"])
    return (;
        states,
        features,
        source_episodes,
        source_steps,
        source_checkpoint,
    )
end

function save_fixed_state_bank(path, bank, config)
    jldsave(
        path;
        states=bank.states,
        features=bank.features,
        source_episodes=bank.source_episodes,
        source_steps=bank.source_steps,
        source_checkpoint=bank.source_checkpoint,
        initial_distribution=config.initial_distribution_name,
    )
    return path
end

function residual_output_columns(config, checkpoints, args)
    summary_columns = propertynames(residual_statistics(
        checkpoint_iteration(first(checkpoints)),
        first(checkpoints),
        (
            value_predictions=[0.0],
            backups=[0.0],
            search_times=[0.0],
            steps=[1],
            epsilon=config.epsilon_schedule(1),
        ),
        args,
        args.state_mode,
    ))
    sample_columns = [
        "iteration",
        "sample_index",
        "source_checkpoint",
        "source_episode",
        "source_step",
        "is_root",
        "value_prediction",
        "search_backup",
        "residual",
        "absolute_residual",
        "squared_residual",
        "search_seconds",
        ["state_$(index)" for index in 1:config.state_dim]...,
    ]
    return summary_columns, sample_columns
end

function run_serial_residual_sweep(
        config,
        oracle,
        checkpoints,
        bank,
        bank_checkpoint,
        args,
    )
    mkpath(args.output_dir)
    summary_columns, sample_columns =
        residual_output_columns(config, checkpoints, args)
    summary_path = joinpath(args.output_dir, "summary.csv")
    samples_path = joinpath(args.output_dir, "samples.csv")
    metadata_path = joinpath(args.output_dir, "metadata.csv")
    write_metadata(
        metadata_path,
        config,
        args,
        checkpoints,
        bank_checkpoint,
    )

    println(
        "[shapley-residual] experiment=$(config.name) ",
        "mode=$(args.state_mode) checkpoints=$(length(checkpoints)) ",
        "samples/checkpoint=$(args.samples_per_checkpoint) ",
        "queries=$(args.tree_queries) depth=$(args.max_depth)",
    )
    println(
        "[shapley-residual] iterations=",
        join(checkpoint_iteration.(checkpoints), ','),
    )
    flush(stdout)

    open(summary_path, "w") do summary_io
        open(samples_path, "w") do samples_io
            write_csv_header(summary_io, summary_columns)
            println(samples_io, join(sample_columns, ','))
            flush(samples_io)
            for checkpoint in checkpoints
                iteration = checkpoint_iteration(checkpoint)
                samples = args.state_mode == "fixed" ?
                    fixed_bank_samples(
                        config,
                        oracle,
                        checkpoint,
                        bank,
                        args,
                    ) :
                    on_policy_samples(
                        config,
                        oracle,
                        checkpoint,
                        args,
                    )
                all(
                    length(features) == config.state_dim
                    for features in samples.features
                ) || error("Unexpected encoded state dimension")
                summary = residual_statistics(
                    iteration,
                    checkpoint,
                    samples,
                    args,
                    args.state_mode,
                )
                write_csv_row(summary_io, summary, summary_columns)
                write_sample_rows(
                    samples_io,
                    iteration,
                    samples,
                    config.state_dim,
                )
                println(
                    "[shapley-residual] iter=$(iteration) ",
                    "L1=$(round(summary.residual_l1; digits=6)) ",
                    "L2=$(round(summary.residual_l2; digits=6)) ",
                    "Linf=$(round(summary.residual_linf; digits=6)) ",
                    "signed=$(round(summary.residual_mean; digits=6))",
                )
                flush(stdout)
            end
        end
    end
    println("[shapley-residual] wrote $(summary_path)")
    println("[shapley-residual] wrote $(samples_path)")
    println("[shapley-residual] wrote $(metadata_path)")
    return (; summary_path, samples_path, metadata_path)
end

function worker_command(config, args, iterations, output_dir, bank_path)
    command = [
        Base.julia_cmd().exec;
        "--project=$(dirname(Base.active_project()))";
        config.runner_file;
        "--worker";
        "--workers"; "0";
        "--state-mode"; args.state_mode;
        "--checkpoint-iterations"; join(iterations, ',');
        "--samples-per-checkpoint"; string(args.samples_per_checkpoint);
        "--tree-queries"; string(args.tree_queries);
        "--max-depth"; string(args.max_depth);
        "--sim-depth"; string(args.sim_depth);
        "--seed"; string(args.seed);
        "--output-dir"; output_dir;
    ]
    isempty(bank_path) || append!(
        command,
        ["--state-bank-path", bank_path],
    )
    return addenv(
        Cmd(command),
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
    )
end

function completed_checkpoint_count(summary_paths)
    return sum(summary_paths) do path
        isfile(path) || return 0
        return max(countlines(path) - 1, 0)
    end
end

function merge_worker_csv(output_path, input_paths; sample_rows=false)
    all(isfile, input_paths) ||
        error("Missing worker CSV while merging $(output_path)")
    contents = readlines.(input_paths)
    headers = unique(first(lines) for lines in contents)
    length(headers) == 1 ||
        error("Worker CSV headers do not match for $(output_path)")
    rows = reduce(vcat, [lines[2:end] for lines in contents])
    sort!(rows; by=line -> begin
        fields = split(line, ','; limit=3)
        iteration = parse(Int, fields[1])
        sample_index = sample_rows ? parse(Int, fields[2]) : 0
        return iteration, sample_index
    end)
    open(output_path, "w") do io
        println(io, only(headers))
        foreach(line -> println(io, line), rows)
    end
    return output_path
end

function run_parallel_residual_sweep(
        config,
        checkpoints,
        bank,
        bank_checkpoint,
        args,
    )
    mkpath(args.output_dir)
    bank_path = ""
    if !isnothing(bank)
        bank_path = save_fixed_state_bank(
            joinpath(args.output_dir, "state_bank.jld2"),
            bank,
            config,
        )
    end
    metadata_path = write_metadata(
        joinpath(args.output_dir, "metadata.csv"),
        config,
        args,
        checkpoints,
        bank_checkpoint,
    )
    worker_count = min(args.workers, length(checkpoints))
    groups = [
        checkpoints[worker:worker_count:end]
        for worker in 1:worker_count
    ]
    println(
        "[shapley-residual] parallel workers=$(worker_count) ",
        "checkpoints=$(length(checkpoints))",
    )
    flush(stdout)

    return mktempdir() do temp_dir
        processes = Base.Process[]
        log_ios = IO[]
        log_paths = String[]
        summary_paths = String[]
        sample_paths = String[]
        for (worker, group) in enumerate(groups)
            worker_dir = joinpath(temp_dir, "worker$(worker)")
            mkpath(worker_dir)
            log_path = joinpath(temp_dir, "worker$(worker).log")
            log_io = open(log_path, "w")
            iterations = checkpoint_iteration.(group)
            command = worker_command(
                config,
                args,
                iterations,
                worker_dir,
                bank_path,
            )
            process = run(
                pipeline(command; stdout=log_io, stderr=log_io);
                wait=false,
            )
            push!(processes, process)
            push!(log_ios, log_io)
            push!(log_paths, log_path)
            push!(summary_paths, joinpath(worker_dir, "summary.csv"))
            push!(sample_paths, joinpath(worker_dir, "samples.csv"))
        end

        progress = Progress(
            length(checkpoints);
            desc="parallel checkpoint sweep",
        )
        last_completed = 0
        while any(Base.process_running, processes)
            completed = completed_checkpoint_count(summary_paths)
            completed > last_completed && update!(progress, completed)
            last_completed = completed
            sleep(0.25)
        end
        foreach(wait, processes)
        foreach(close, log_ios)
        completed = completed_checkpoint_count(summary_paths)
        completed > last_completed && update!(progress, completed)
        finish!(progress)

        failed = findall(process -> !success(process), processes)
        if !isempty(failed)
            for worker in failed
                println(stderr, "----- worker $(worker) log -----")
                print(stderr, read(log_paths[worker], String))
            end
            error("Residual worker processes failed: $(join(failed, ", "))")
        end

        summary_path = merge_worker_csv(
            joinpath(args.output_dir, "summary.csv"),
            summary_paths,
        )
        samples_path = merge_worker_csv(
            joinpath(args.output_dir, "samples.csv"),
            sample_paths;
            sample_rows=true,
        )
        println("[shapley-residual] wrote $(summary_path)")
        println("[shapley-residual] wrote $(samples_path)")
        println("[shapley-residual] wrote $(metadata_path)")
        isempty(bank_path) ||
            println("[shapley-residual] wrote $(bank_path)")
        return (; summary_path, samples_path, metadata_path)
    end
end

"""
    run_approximate_shapley_residual(config, raw_args=ARGS)

Measure the empirical fixed-point residual

    (approximate depth-D search backup of V_k)(s) - V_k(s)

across a checkpoint sequence. The default `--state-mode fixed` first gathers
one fixed state bank through ordinary no-transfer self-play, then reuses that
bank for every checkpoint. `--state-mode on-policy` instead gathers a fresh
self-play batch from each checkpoint and therefore mirrors training more
literally while allowing the evaluation distribution to move with k.
"""
function run_approximate_shapley_residual(config, raw_args=ARGS)
    args = parse_residual_args(config, raw_args)
    explicit = explicit_checkpoints(
        args.all_paths,
        args.checkpoint_iterations,
    )
    checkpoints = isnothing(explicit) ?
        select_checkpoints(
            args.all_paths;
            every=args.every,
            first_iteration=args.first_iteration,
            last_iteration=args.last_iteration,
        ) :
        explicit
    args.test && (checkpoints = unique([first(checkpoints), last(checkpoints)]))
    isfile(config.oracle_file) ||
        error("Missing oracle architecture file: $(config.oracle_file)")
    oracle = AZ.load_oracle(config.oracle_file)
    oracle isa AZ.FittedRegretModel ||
        error("Expected FittedRegretModel, got $(typeof(oracle))")

    mkpath(args.output_dir)
    bank_checkpoint = args.state_mode == "fixed" ?
        resolve_checkpoint(args.all_paths, args.bank_checkpoint_spec) : nothing
    bank = if args.state_mode != "fixed"
        nothing
    elseif !isempty(args.state_bank_path)
        load_fixed_state_bank(abspath(args.state_bank_path))
    else
        collect_fixed_state_bank(config, oracle, bank_checkpoint, args)
    end

    if args.worker || args.workers <= 1 || length(checkpoints) == 1
        saved_bank_path = ""
        if !isnothing(bank) && isempty(args.state_bank_path)
            saved_bank_path = save_fixed_state_bank(
                joinpath(args.output_dir, "state_bank.jld2"),
                bank,
                config,
            )
        end
        result = run_serial_residual_sweep(
            config,
            oracle,
            checkpoints,
            bank,
            bank_checkpoint,
            args,
        )
        isempty(saved_bank_path) ||
            println("[shapley-residual] wrote $(saved_bank_path)")
        return result
    end
    return run_parallel_residual_sweep(
        config,
        checkpoints,
        bank,
        bank_checkpoint,
        args,
    )
end
