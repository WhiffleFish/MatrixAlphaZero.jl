using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Distributed
using Distributions
using Flux
using JLD2
using MarkovGames
using MatrixAlphaZero
using POMDPTools
using Printf
using ProgressMeter
using Random
using SDAGames.SNRGame
using SDAGames.SatelliteDynamics
using SHA
using Statistics

const AZ = MatrixAlphaZero
const SEARCH_NAME = "rm_plus_no_transfer_train_mean_leo"
const DEFAULT_OUTPUT = joinpath(@__DIR__, "regret_fit_dataset_final_iter.jld2")
const DEFAULT_TARGET_STEPS = 65_536

include(joinpath(@__DIR__, "initial_state.jl"))

function option_value(args, name, default, parse_value=identity)
    idx = findfirst(==(name), args)
    isnothing(idx) && return default
    idx < length(args) || error("Missing value after $(name)")
    return parse_value(args[idx + 1])
end

function checkpoint_iteration(path)
    matched = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(matched) && error("Invalid checkpoint name: $(path)")
    return parse(Int, only(matched.captures))
end

function select_checkpoint(spec::AbstractString)
    models_dir = joinpath(@__DIR__, "models_$(SEARCH_NAME)")
    checkpoints = filter(
        path -> occursin(r"oracle\d+\.jld2$", basename(path)),
        readdir(models_dir; join=true),
    )
    isempty(checkpoints) && error("No checkpoints found in $(models_dir)")
    sort!(checkpoints; by=checkpoint_iteration)
    spec == "latest" && return last(checkpoints)
    requested = parse(Int, spec)
    matches = filter(path -> checkpoint_iteration(path) == requested, checkpoints)
    isempty(matches) && error("No checkpoint for iteration $(requested)")
    return only(matches)
end

function sha256_file(path)
    return bytes2hex(open(SHA.sha256, path))
end

function format_duration(seconds::Real)
    isfinite(seconds) || return "unknown"
    total_seconds = max(0, round(Int, seconds))
    hours, remainder = divrem(total_seconds, 3600)
    minutes, seconds = divrem(remainder, 60)
    return hours > 0 ? "$(hours)h $(minutes)m $(seconds)s" : "$(minutes)m $(seconds)s"
end

function log_collection_progress(collected, target, episodes, started_at, chunk_seconds)
    elapsed = time() - started_at
    rate = collected / max(elapsed, eps(Float64))
    eta = rate > 0 ? (target - collected) / rate : Inf
    @printf(
        stderr,
        "[regret-data] samples=%d/%d (%.1f%%) episodes=%d elapsed=%s rate=%.2f samples/s eta=%s last_chunk=%s\n",
        collected,
        target,
        100 * collected / target,
        episodes,
        format_duration(elapsed),
        rate,
        format_duration(eta),
        format_duration(chunk_seconds),
    )
    flush(stderr)
end

function split_by_episode(episode_ids, seed)
    episodes = unique(episode_ids)
    length(episodes) >= 3 || error(
        "Need at least three episodes for train/validation/test splitting; got $(length(episodes))",
    )
    shuffled = shuffle(MersenneTwister(seed), episodes)
    n_test = max(1, round(Int, 0.10 * length(shuffled)))
    n_validation = max(1, round(Int, 0.10 * length(shuffled)))
    n_train = length(shuffled) - n_validation - n_test
    n_train >= 1 || error("Not enough episodes for a nonempty training split")
    train_episodes = Set(shuffled[1:n_train])
    validation_episodes = Set(shuffled[(n_train + 1):(n_train + n_validation)])
    test_episodes = Set(shuffled[(n_train + n_validation + 1):end])
    train_indices = findall(in(train_episodes), episode_ids)
    validation_indices = findall(in(validation_episodes), episode_ids)
    test_indices = findall(in(test_episodes), episode_ids)
    return train_indices, validation_indices, test_indices
end

function main(args=ARGS)
    test = "--test" in args
    force = "--force" in args
    output = abspath(option_value(args, "--output", DEFAULT_OUTPUT, String))
    checkpoint_spec = option_value(args, "--checkpoint", "latest", String)
    workers_requested = option_value(args, "--workers", 8, x -> parse(Int, x))
    target_steps = option_value(args, "--steps", DEFAULT_TARGET_STEPS, x -> parse(Int, x))
    seed = option_value(args, "--seed", 20260722, x -> parse(Int, x))
    chunk_episodes = option_value(args, "--chunk-episodes", 8, x -> parse(Int, x))
    search_epsilon = option_value(args, "--search-epsilon", 0.1, x -> parse(Float64, x))
    action_epsilon = option_value(args, "--action-epsilon", 0.3, x -> parse(Float64, x))

    tree_queries = 500
    max_depth = 5
    sim_depth = 50
    gae_lambda = 0.95
    if test
        target_steps = min(target_steps, 48)
        tree_queries = 4
        max_depth = 2
        sim_depth = 8
        workers_requested = min(workers_requested, 2)
        chunk_episodes = 8
    end

    target_steps > 0 || error("--steps must be positive")
    workers_requested >= 0 || error("--workers must be nonnegative")
    chunk_episodes > 0 || error("--chunk-episodes must be positive")
    0 <= search_epsilon <= 1 || error("--search-epsilon must be in [0, 1]")
    0 <= action_epsilon <= 1 || error("--action-epsilon must be in [0, 1]")
    if isfile(output) && !force
        error("Refusing to overwrite existing dataset $(output); pass --force to replace it")
    end

    checkpoint = select_checkpoint(checkpoint_spec)
    checkpoint_iter = checkpoint_iteration(checkpoint)
    oracle_file = joinpath(@__DIR__, "oracle_$(SEARCH_NAME).jld2")
    oracle = AZ.load_oracle(oracle_file)
    oracle isa AZ.FittedRegretModel || error(
        "Expected FittedRegretModel architecture, got $(typeof(oracle))",
    )
    Flux.loadmodel!(oracle, checkpoint)

    game = SNRGameSimple(altitude_bounds=(100e3, 2e7))
    initial_distribution = core_initialstate_distribution(game)
    search = AZ.MCTSSearch(;
        oracle,
        tree_queries,
        max_depth,
        max_time=Inf,
        search_style=AZ.RegretMatchingSearch(; backup=:mean, method=AZ.Plus()),
        value_target=:search,
        ϵ=_ -> search_epsilon,
        prior_scale=0.0,
    )

    println(
        "Collecting root-only final-iterate regret targets: ",
        "samples=$(target_steps) workers=$(workers_requested) chunk_episodes=$(chunk_episodes)",
    )
    println(
        "Search configuration: queries=$(tree_queries) depth=$(max_depth) ",
        "backup=mean search_epsilon=$(search_epsilon) action_epsilon=$(action_epsilon)",
    )
    println("Checkpoint: $(checkpoint)")
    println("Output: $(output)")
    flush(stdout)

    added_workers = Int[]
    if workers_requested > 0
        println("[regret-data] launching $(workers_requested) worker processes")
        flush(stdout)
        added_workers = addprocs(workers_requested)
        for pid in added_workers
            remotecall_wait(pid) do
                Core.eval(Main, quote
                    using Distributions
                    using Flux
                    using MarkovGames
                    using MatrixAlphaZero
                    using POMDPTools
                    using Random
                    using SDAGames.SNRGame
                    using SDAGames.SatelliteDynamics
                    const AZ = MatrixAlphaZero
                end)
                nothing
            end
        end
        println("[regret-data] workers ready: $(join(added_workers, ','))")
        flush(stdout)
    end
    pool = isempty(added_workers) ? nothing : CachingPool(added_workers)

    states = Vector{Float32}[]
    regret_1 = Vector{Float64}[]
    regret_2 = Vector{Float64}[]
    episode_ids = Int[]
    episode_seeds = Int[]
    collected_steps = 0
    episode_id = 0
    progress = Progress(
        target_steps;
        desc="Root regret targets: ",
        showspeed=true,
        dt=1.0,
    )
    started_at = time()
    println("[regret-data] beginning root-state collection")
    flush(stdout)
    try
        while collected_steps < target_steps
            chunk_started_at = time()
            seeds = collect(
                (seed + 1_000_000 + episode_id + 1):
                (seed + 1_000_000 + episode_id + chunk_episodes),
            )
            generate = episode_seed -> begin
                Random.seed!(episode_seed)
                initial_state = rand(initial_distribution)
                AZ.mcts_regret_sim(
                    search,
                    game,
                    initial_state;
                    search_ϵ=search_epsilon,
                    action_ϵ=action_epsilon,
                    sim_depth,
                    gae_lambda,
                    progress=false,
                )
            end
            histories = if isnothing(pool)
                map(generate, seeds)
            else
                pmap(generate, pool, seeds; batch_size=1)
            end
            for (episode_seed, history) in zip(seeds, histories)
                collected_steps >= target_steps && break
                episode_id += 1
                remaining = target_steps - collected_steps
                n = min(remaining, length(history.s))
                iszero(n) && continue
                append!(states, history.s[1:n])
                append!(regret_1, history.regret[1][1:n])
                append!(regret_2, history.regret[2][1:n])
                append!(episode_ids, fill(episode_id, n))
                push!(episode_seeds, episode_seed)
                collected_steps += n
                ProgressMeter.update!(progress, collected_steps)
            end
            log_collection_progress(
                collected_steps,
                target_steps,
                episode_id,
                started_at,
                time() - chunk_started_at,
            )
        end
    finally
        ProgressMeter.finish!(progress)
        isempty(added_workers) || rmprocs(added_workers)
    end

    X = Float32.(reduce(hcat, states))
    regret_p1 = Float32.(reduce(hcat, regret_1))
    regret_p2 = Float32.(reduce(hcat, regret_2))
    checkpoint_regret = AZ.regret(oracle, X)
    checkpoint_strategy = AZ.strategy(oracle, X)
    checkpoint_value = vec(AZ.value(oracle, X))
    train_indices, validation_indices, test_indices = split_by_episode(
        episode_ids,
        seed + 2_000_000,
    )

    metadata = Dict{String,Any}(
        "experiment" => "sda-2026-07-21",
        "distribution" => CORE_DISTRIBUTION_NAME,
        "checkpoint" => abspath(checkpoint),
        "checkpoint_iteration" => checkpoint_iter,
        "checkpoint_sha256" => sha256_file(checkpoint),
        "architecture_file" => abspath(oracle_file),
        "architecture_sha256" => sha256_file(oracle_file),
        "seed" => seed,
        "split_seed" => seed + 2_000_000,
        "target_steps" => target_steps,
        "episodes" => episode_id,
        "episode_seeds" => episode_seeds,
        "tree_queries" => tree_queries,
        "max_depth" => max_depth,
        "sim_depth" => sim_depth,
        "search_epsilon" => search_epsilon,
        "action_epsilon" => action_epsilon,
        "trajectory_policy" => "local_rm_plus_average_strategy_with_separate_action_epsilon",
        "sample_scope" => "environment_trajectory_search_roots_only",
        "includes_nonroot_tree_states" => false,
        "checkpoint_strategy_used_as_search_prior" => false,
        "backup" => "mean",
        "method" => "rm_plus",
        "prior_scale" => 0.0,
        "value_target" => "search",
        "gae_lambda" => gae_lambda,
        "test_mode" => test,
        "state_dim" => size(X, 1),
        "action_counts" => [size(regret_p1, 1), size(regret_p2, 1)],
    )

    mkpath(dirname(output))
    jldsave(
        output;
        states=X,
        regret_p1,
        regret_p2,
        episode_ids,
        train_indices,
        validation_indices,
        test_indices,
        checkpoint_regret_p1=Float32.(checkpoint_regret[1]),
        checkpoint_regret_p2=Float32.(checkpoint_regret[2]),
        checkpoint_strategy_p1=Float32.(checkpoint_strategy[1]),
        checkpoint_strategy_p2=Float32.(checkpoint_strategy[2]),
        checkpoint_value=Float32.(checkpoint_value),
        metadata,
    )

    println("Wrote frozen regret dataset: $(output)")
    println(
        "samples=$(size(X, 2)) episodes=$(episode_id) ",
        "splits=$(length(train_indices))/$(length(validation_indices))/$(length(test_indices))",
    )
    for (player, target) in enumerate((regret_p1, regret_p2))
        println(
            "p$(player): zero_fraction=$(round(mean(iszero, target); digits=4)) ",
            "mean=$(round(mean(target); digits=6)) ",
            "q99=$(round(quantile(vec(target), 0.99); digits=6)) ",
            "max=$(round(maximum(target); digits=6))",
        )
    end
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
