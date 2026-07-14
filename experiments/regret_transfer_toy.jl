using Pkg
Pkg.activate("experiments")

# A small sequential benchmark for regret transfer.
#
# The game has three modes and a finite horizon. At every step the players play
# a different skewed RPS matrix, and their joint action selects the next mode.
# Thus continuation values alter the effective matrix game at the current state.
#
# A long source search at every state produces the targets that a perfectly
# fitted regret model would predict:
#
#     R_hat(s) = R_source(s) / sqrt(T_source)
#     S_hat(s) = S_source(s) / T_source
#
# Short online searches then run through the actual multi-step tree with and
# without those priors. Exploitability is computed exactly by recursive best
# response over the full horizon.
#
# Example:
#   julia --project=experiments experiments/regret_transfer_toy.jl

using DelimitedFiles
using LinearAlgebra
using MarkovGames
using MatrixAlphaZero
using POMDPs
using POMDPTools
using Printf
using Random
using Statistics

const AZ = MatrixAlphaZero
const SCRIPT_DIR = @__DIR__

struct SequentialState
    step::Int
    mode::Int
end

struct SequentialRPS <: MG{SequentialState,NTuple{2,Int}}
    horizon::Int
end

const BASE_PAYOFF = [
     0.0       -1.0 / 6.0   1.0 / 2.0
     1.0 / 6.0  0.0        -1.0
    -1.0 / 2.0  1.0         0.0
]
const ACTION_PERMUTATIONS = (
    (1, 2, 3),
    (2, 3, 1),
    (3, 1, 2),
)
const MODE_OFFSETS = (-0.12, 0.0, 0.12)

POMDPs.initialstate(::SequentialRPS) = Deterministic(SequentialState(1, 1))
POMDPs.discount(::SequentialRPS) = 1.0
POMDPs.isterminal(game::SequentialRPS, state::SequentialState) = state.step > game.horizon
POMDPs.actions(::SequentialRPS) = (1:3, 1:3)

function state_payoff(::SequentialRPS, state::SequentialState)
    permutation = ACTION_PERMUTATIONS[state.mode]
    scale = 0.45 + 0.10 * (state.step - 1)
    return scale .* BASE_PAYOFF[collect(permutation), collect(permutation)] .+
        MODE_OFFSETS[state.mode]
end

function next_state(state::SequentialState, action::NTuple{2,Int})
    action1, action2 = action
    next_mode = mod1(state.mode + 2 * action1 + action2 + state.step, 3)
    return SequentialState(state.step + 1, next_mode)
end

function POMDPs.gen(
        game::SequentialRPS,
        state::SequentialState,
        action::NTuple{2,Int},
        ::Random.AbstractRNG=Random.default_rng(),
    )
    return (sp=next_state(state, action), r=state_payoff(game, state)[action...])
end

POMDPs.convert_s(::Type{Vector{Float32}}, state::SequentialState, ::SequentialRPS) =
    Float32[state.step, state.mode]

nonterminal_states(game::SequentialRPS) =
    [SequentialState(step, mode) for step in 1:game.horizon for mode in 1:3]

struct TableOracle
    values::Dict{SequentialState,Float32}
    regrets::Dict{SequentialState,NTuple{2,Vector{Float32}}}
    strategies::Dict{SequentialState,NTuple{2,Vector{Float32}}}
end

input_state(input::AbstractVector) =
    SequentialState(round(Int, input[1]), round(Int, input[2]))

AZ.value(oracle::TableOracle, input::AbstractVector) =
    Float32[get(oracle.values, input_state(input), 0.0f0)]

function AZ.value(oracle::TableOracle, input::AbstractMatrix)
    result = Float32[get(oracle.values, input_state(column), 0.0f0)
                     for column in eachcol(input)]
    return reshape(result, 1, :)
end

AZ.state_regret(oracle::TableOracle, ::MG, state::SequentialState) =
    get(oracle.regrets, state, (zeros(Float32, 3), zeros(Float32, 3)))

AZ.state_strategy(oracle::TableOracle, ::MG, state::SequentialState) =
    get(oracle.strategies, state, (fill(1.0f0 / 3, 3), fill(1.0f0 / 3, 3)))

AZ.batch_state_value(oracle::TableOracle, ::MG, states) =
    Float32[get(oracle.values, state, 0.0f0) for state in states]

function supports(n::Int, size::Int)
    size == 1 && return [[index] for index in 1:n]
    size == 2 && return [[i, j] for i in 1:n for j in (i + 1):n]
    size == n && return [collect(1:n)]
    return Vector{Int}[]
end

function solve_zero_sum(payoff::AbstractMatrix; tolerance::Float64=1e-9)
    n1, n2 = size(payoff)
    for support_size in 1:min(n1, n2)
        for support1 in supports(n1, support_size)
            for support2 in supports(n2, support_size)
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

                strategy1 = zeros(n1)
                strategy2 = zeros(n2)
                strategy1[support1] .= solution1[1:support_size]
                strategy2[support2] .= solution2[1:support_size]
                minimum(strategy1) >= -tolerance || continue
                minimum(strategy2) >= -tolerance || continue
                strategy1 .= max.(strategy1, 0.0)
                strategy2 .= max.(strategy2, 0.0)
                strategy1 ./= sum(strategy1)
                strategy2 ./= sum(strategy2)

                row_values = payoff * strategy2
                column_values = transpose(payoff) * strategy1
                value = 0.5 * (maximum(row_values) + minimum(column_values))
                maximum(row_values) <= value + 20 * tolerance || continue
                minimum(column_values) >= value - 20 * tolerance || continue
                return strategy1, strategy2, value
            end
        end
    end
    error("failed to solve matrix game:\n$(payoff)")
end

function equilibrium_tables(game::SequentialRPS)
    values = Dict{SequentialState,Float64}()
    strategies = Dict{SequentialState,NTuple{2,Vector{Float64}}}()
    for step in game.horizon:-1:1
        for mode in 1:3
            state = SequentialState(step, mode)
            q = state_payoff(game, state)
            for action1 in 1:3, action2 in 1:3
                q[action1, action2] += get(
                    values,
                    next_state(state, (action1, action2)),
                    0.0,
                )
            end
            strategy1, strategy2, value = solve_zero_sum(q)
            values[state] = value
            strategies[state] = (strategy1, strategy2)
        end
    end
    return (; values, strategies)
end

function parse_list(::Type{T}, value::AbstractString) where {T}
    entries = filter(!isempty, strip.(split(value, ',')))
    isempty(entries) && error("expected a nonempty comma-separated list")
    return parse.(T, entries)
end

function parse_list(::Type{String}, value::AbstractString)
    entries = String.(filter(!isempty, strip.(split(value, ','))))
    isempty(entries) && error("expected a nonempty comma-separated list")
    return entries
end

function parse_cli(args)
    values = Dict(
        "horizon" => "3",
        "source-updates" => "4096",
        "online-updates" => "4,8,16,32,64,128",
        "prior-masses" => "1,4,16,64,256,1024,4096",
        "corruptions" => "0,0.1,0.25,0.5,0.75,1.0",
        "methods" => "vanilla,plus",
        "trials" => "50",
        "epsilon" => "0.0",
        "seed" => "20260714",
        "output-dir" => joinpath(SCRIPT_DIR, "regret_transfer_noise_results"),
        "test" => "false",
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--test"
            values["test"] = "true"
        elseif startswith(arg, "--")
            parts = split(arg[3:end], '='; limit=2)
            key = first(parts)
            haskey(values, key) || error("unknown option --$(key)")
            if length(parts) == 2
                values[key] = last(parts)
            else
                index < length(args) || error("missing value for --$(key)")
                index += 1
                values[key] = args[index]
            end
        else
            error("unexpected positional argument $(arg)")
        end
        index += 1
    end

    methods = lowercase.(parse_list(String, values["methods"]))
    all(method -> method in ("vanilla", "plus"), methods) ||
        error("--methods must contain only vanilla or plus")

    config = (;
        horizon=parse(Int, values["horizon"]),
        source_updates=parse(Int, values["source-updates"]),
        online_updates=parse_list(Int, values["online-updates"]),
        prior_masses=parse_list(Float64, values["prior-masses"]),
        corruptions=unique(parse_list(Float64, values["corruptions"])),
        methods=unique(methods),
        trials=parse(Int, values["trials"]),
        epsilon=parse(Float64, values["epsilon"]),
        seed=parse(Int, values["seed"]),
        output_dir=abspath(values["output-dir"]),
        test=parse(Bool, values["test"]),
    )
    config.horizon > 1 || error("--horizon must be at least 2")
    config.source_updates > 0 || error("--source-updates must be positive")
    all(>(0), config.online_updates) || error("--online-updates must be positive")
    all(>(0), config.prior_masses) || error("--prior-masses must be positive")
    all(corruption -> 0 <= corruption <= 1, config.corruptions) ||
        error("--corruptions must be in [0, 1]")
    config.trials > 0 || error("--trials must be positive")
    0 <= config.epsilon <= 1 || error("--epsilon must be in [0, 1]")

    return config.test ? merge(config, (;
        source_updates=256,
        online_updates=[4, 16],
        prior_masses=[4.0, 64.0, 256.0],
        corruptions=[0.0, 0.5, 1.0],
        trials=5,
        output_dir=joinpath(config.output_dir, "smoke"),
    )) : config
end

method_object(name::String) = name == "plus" ? AZ.Plus() : AZ.Vanilla()

function empty_oracle(game::SequentialRPS, equilibrium)
    values = Dict(state => Float32(value) for (state, value) in equilibrium.values)
    regrets = Dict{SequentialState,NTuple{2,Vector{Float32}}}()
    strategies = Dict{SequentialState,NTuple{2,Vector{Float32}}}()
    return TableOracle(values, regrets, strategies)
end

function matched_scale_noise(rng, target)
    noise = randn(rng, length(target))
    target_norm = norm(target)
    noise_norm = norm(noise)
    iszero(target_norm) && return zeros(length(target))
    return target_norm .* noise ./ noise_norm
end

function corrupted_source_oracle(equilibrium, source, corruption::Float64, seed::Int)
    rng = MersenneTwister(seed)
    values = Dict(state => Float32(value) for (state, value) in equilibrium.values)
    regrets = Dict{SequentialState,NTuple{2,Vector{Float32}}}()
    strategies = Dict{SequentialState,NTuple{2,Vector{Float32}}}()
    for state in sort!(collect(keys(source)); by=state -> (state.step, state.mode))
        record = source[state]
        true_regret = (
            record.regret[1] ./ sqrt(record.mass),
            record.regret[2] ./ sqrt(record.mass),
        )
        random_regret = (
            matched_scale_noise(rng, true_regret[1]),
            matched_scale_noise(rng, true_regret[2]),
        )
        regrets[state] = (
            Float32.((1 - corruption) .* true_regret[1] .+ corruption .* random_regret[1]),
            Float32.((1 - corruption) .* true_regret[2] .+ corruption .* random_regret[2]),
        )

        random_strategy = (rand(rng, 3), rand(rng, 3))
        random_strategy = (random_strategy[1] ./ sum(random_strategy[1]),
                           random_strategy[2] ./ sum(random_strategy[2]))
        strategies[state] = (
            Float32.((1 - corruption) .* record.strategy[1] .+ corruption .* random_strategy[1]),
            Float32.((1 - corruption) .* record.strategy[2] .+ corruption .* random_strategy[2]),
        )
    end
    return TableOracle(values, regrets, strategies)
end

function run_state_search(
        game::SequentialRPS,
        oracle,
        state::SequentialState,
        method_name::String,
        updates::Int,
        seed::Int;
        prior_mass::Float64=0.0,
        epsilon::Float64=0.0,
        max_depth::Int=game.horizon - state.step + 1,
    )
    transfer_enabled = prior_mass > 0
    search = AZ.MCTSSearch(;
        oracle,
        tree_queries=updates + 1,
        max_depth,
        search_style=AZ.RegretMatchingSearch(; method=method_object(method_name)),
        transfer_weight=transfer_enabled ? 1.0 : 0.0,
        τ=prior_mass,
        ϵ=_ -> epsilon,
    )
    Random.seed!(seed)
    (strategy1, strategy2, value), info = AZ.search_info(search, game, state; ϵ=epsilon)
    return (;
        strategy=(Float64.(strategy1), Float64.(strategy2)),
        value,
        tree=info.tree,
    )
end

function make_source(game, equilibrium, method_name::String, updates::Int, seed::Int)
    oracle = empty_oracle(game, equilibrium)
    source = Dict{SequentialState,Any}()
    for state in nonterminal_states(game)
        state_seed = seed + 100 * state.step + state.mode
        result = run_state_search(
            game,
            oracle,
            state,
            method_name,
            updates,
            state_seed;
            max_depth=1,
        )
        tree = result.tree
        mass1 = sum(tree.policy_sum[1][1])
        mass2 = sum(tree.policy_sum[2][1])
        isapprox(mass1, updates; atol=1e-8) || error("invalid source strategy mass")
        isapprox(mass2, updates; atol=1e-8) || error("invalid source strategy mass")
        source[state] = (;
            mass=Float64(updates),
            regret=(copy(tree.regret[1][1]), copy(tree.regret[2][1])),
            strategy=result.strategy,
        )
    end
    return source
end

function reconstruction_metrics(game, method_name::String, source, oracle)
    predicted_regret = Float64[]
    target_regret = Float64[]
    predicted_strategy = Float64[]
    target_strategy = Float64[]
    strategy_tvs = Float64[]
    for state in nonterminal_states(game)
        record = source[state]
        search = AZ.MCTSSearch(;
            oracle,
            tree_queries=0,
            max_depth=1,
            search_style=AZ.RegretMatchingSearch(; method=method_object(method_name)),
            transfer_weight=1.0,
            τ=record.mass,
        )
        tree = AZ.Tree(search, game, state)
        AZ.expand_s!(tree, 1, game, oracle)
        AZ.warmstart_node!(search, tree, 1, game)
        for player in 1:2
            append!(predicted_regret, tree.regret[player][1])
            append!(target_regret, record.regret[player])
            append!(predicted_strategy, tree.policy_sum[player][1])
            append!(target_strategy, record.mass .* record.strategy[player])
            reconstructed_policy = AZ.normalized_or_uniform(tree.policy_sum[player][1])
            push!(strategy_tvs, 0.5 * sum(abs.(reconstructed_policy .- record.strategy[player])))
        end
    end
    regret_relative_error = norm(predicted_regret .- target_regret) / max(norm(target_regret), eps())
    strategy_relative_error = norm(predicted_strategy .- target_strategy) / max(norm(target_strategy), eps())
    regret_cosine = dot(predicted_regret, target_regret) /
        max(norm(predicted_regret) * norm(target_regret), eps())
    return (;
        regret_relative_error,
        strategy_relative_error,
        regret_cosine,
        strategy_tv=mean(strategy_tvs),
    )
end

function exact_policy_values(game, policies, state::SequentialState)
    lower_cache = Dict{SequentialState,Float64}()
    upper_cache = Dict{SequentialState,Float64}()

    function lower_value(current)
        isterminal(game, current) && return 0.0
        haskey(lower_cache, current) && return lower_cache[current]
        strategy1 = policies[current][1]
        payoff = state_payoff(game, current)
        action2_values = zeros(3)
        for action2 in 1:3, action1 in 1:3
            child = next_state(current, (action1, action2))
            action2_values[action2] += strategy1[action1] *
                (payoff[action1, action2] + lower_value(child))
        end
        return lower_cache[current] = minimum(action2_values)
    end

    function upper_value(current)
        isterminal(game, current) && return 0.0
        haskey(upper_cache, current) && return upper_cache[current]
        strategy2 = policies[current][2]
        payoff = state_payoff(game, current)
        action1_values = zeros(3)
        for action1 in 1:3, action2 in 1:3
            child = next_state(current, (action1, action2))
            action1_values[action1] += strategy2[action2] *
                (payoff[action1, action2] + upper_value(child))
        end
        return upper_cache[current] = maximum(action1_values)
    end

    lower = lower_value(state)
    upper = upper_value(state)
    return (; lower, upper, exploitability=upper - lower)
end

function search_policy(
        game,
        oracle,
        method_name,
        updates,
        seed;
        prior_mass,
        epsilon,
    )
    policies = Dict{SequentialState,NTuple{2,Vector{Float64}}}()
    for state in nonterminal_states(game)
        state_seed = seed + 100 * state.step + state.mode
        result = run_state_search(
            game,
            oracle,
            state,
            method_name,
            updates,
            state_seed;
            prior_mass,
            epsilon,
        )
        policies[state] = result.strategy
    end
    return policies
end

function write_csv(path::String, header, rows)
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] .= header
    for (index, row) in enumerate(rows)
        table[index + 1, :] .= row
    end
    mkpath(dirname(path))
    writedlm(path, table, ',')
    return path
end

stderr(values) = length(values) > 1 ? std(values) / sqrt(length(values)) : 0.0

function summarize(trials)
    grouped = Dict{Tuple{String,Float64,Int,Float64},Vector{Any}}()
    for trial in trials
        key = (trial.method, trial.corruption, trial.online_updates, trial.prior_mass)
        push!(get!(grouped, key, Any[]), trial)
    end

    baseline = Dict{Tuple{String,Float64,Int,Int},Float64}()
    for trial in trials
        iszero(trial.prior_mass) || continue
        baseline[(trial.method, trial.corruption, trial.online_updates, trial.trial)] =
            trial.exploitability
    end

    rows = NamedTuple[]
    for key in sort!(collect(keys(grouped)))
        group = grouped[key]
        gaps = Float64[trial.exploitability for trial in group]
        improvements = Float64[
            baseline[(trial.method, trial.corruption, trial.online_updates, trial.trial)] -
                trial.exploitability
            for trial in group
        ]
        push!(rows, (;
            method=key[1],
            corruption=key[2],
            online_updates=key[3],
            prior_mass=key[4],
            trials=length(group),
            exploitability_mean=mean(gaps),
            exploitability_stderr=stderr(gaps),
            exploitability_p90=quantile(gaps, 0.9),
            paired_improvement_mean=mean(improvements),
            paired_improvement_stderr=stderr(improvements),
            paired_win_rate=mean(>(0), improvements),
        ))
    end
    return rows
end

function print_results(source_rows, fit_rows, summary_rows)
    println("\nSource priors")
    for source in source_rows
        @printf(
            "  %-7s global exploitability=%.6f\n",
            source.method,
            source.exploitability,
        )
    end

    println("\nFitted-prior quality")
    for fit in fit_rows
        @printf(
            "  %-7s corruption=%.2f R-relative=%.3f S-relative=%.3f R-cosine=% .3f S-TV=%.3f\n",
            fit.method,
            fit.corruption,
            fit.regret_relative_error,
            fit.strategy_relative_error,
            fit.regret_cosine,
            fit.strategy_tv,
        )
    end

    println("\nBest transferred configuration at each corruption and online budget")
    for method in sort!(unique(row.method for row in summary_rows))
        corruptions = sort!(unique(row.corruption for row in summary_rows if row.method == method))
        for corruption in corruptions
            budgets = sort!(unique(
                row.online_updates for row in summary_rows
                if row.method == method && row.corruption == corruption
            ))
            for budget in budgets
            rows = filter(
                row -> row.method == method && row.corruption == corruption &&
                    row.online_updates == budget,
                summary_rows,
            )
            baseline = only(filter(row -> iszero(row.prior_mass), rows))
            best = argmin(
                row -> row.exploitability_mean,
                filter(row -> row.prior_mass > 0, rows),
            )
            @printf(
                "  %-7s c=%.2f T=%3d: base %.5f | mass=%6.0f %.5f | gain %.5f +/- %.5f\n",
                method,
                corruption,
                budget,
                baseline.exploitability_mean,
                best.prior_mass,
                best.exploitability_mean,
                best.paired_improvement_mean,
                best.paired_improvement_stderr,
            )
            end
        end
    end

    clear_wins = filter(summary_rows) do row
        row.prior_mass > 0 &&
            row.paired_improvement_mean > 0 &&
            row.paired_improvement_mean > 2 * row.paired_improvement_stderr
    end
    if isempty(clear_wins)
        println("\nRESULT: no statistically clear transfer win found.")
    else
        best = argmax(row -> row.paired_improvement_mean, clear_wins)
        @printf(
            "\nRESULT: PASS. Best gain is %.5f at method=%s, corruption=%.2f, online updates=%d, prior mass=%.0f.\n",
            best.paired_improvement_mean,
            best.method,
            best.corruption,
            best.online_updates,
            best.prior_mass,
        )
    end
    return nothing
end

function main(args)
    config = parse_cli(args)
    game = SequentialRPS(config.horizon)
    initial = rand(initialstate(game))
    equilibrium = equilibrium_tables(game)
    equilibrium_values = exact_policy_values(game, equilibrium.strategies, initial)
    uniform_policies = Dict(
        state => (fill(1 / 3, 3), fill(1 / 3, 3)) for state in nonterminal_states(game)
    )
    uniform_values = exact_policy_values(game, uniform_policies, initial)
    @printf("Game: %d steps, %d nonterminal states\n", game.horizon, length(nonterminal_states(game)))
    @printf("Uniform exploitability: %.6f\n", uniform_values.exploitability)
    @printf("Nash exploitability:    %.6e\n", equilibrium_values.exploitability)

    sources = Dict{String,Any}()
    source_rows = NamedTuple[]
    source_state_rows = Vector{Vector{Any}}()
    for (offset, method_name) in enumerate(config.methods)
        source = make_source(
            game,
            equilibrium,
            method_name,
            config.source_updates,
            config.seed - 1000 - offset,
        )
        source_policies = Dict(state => record.strategy for (state, record) in source)
        source_values = exact_policy_values(game, source_policies, initial)
        sources[method_name] = source
        push!(source_rows, (;
            method=method_name,
            exploitability=source_values.exploitability,
        ))
        for state in nonterminal_states(game)
            record = source[state]
            push!(source_state_rows, Any[
                method_name, state.step, state.mode, record.mass,
                record.strategy[1]..., record.strategy[2]...,
                record.regret[1]..., record.regret[2]...,
            ])
        end
    end

    oracles = Dict{Tuple{String,Float64},TableOracle}()
    fit_rows = NamedTuple[]
    for (method_offset, method_name) in enumerate(config.methods)
        for corruption in sort!(unique(config.corruptions))
            corruption_seed = config.seed + 100_000 * method_offset
            oracle = corrupted_source_oracle(
                equilibrium,
                sources[method_name],
                corruption,
                corruption_seed,
            )
            metrics = reconstruction_metrics(game, method_name, sources[method_name], oracle)
            if iszero(corruption)
                metrics.regret_relative_error < 1e-5 ||
                    error("exact regret reconstruction failed for $(method_name)")
                metrics.strategy_relative_error < 1e-5 ||
                    error("exact strategy reconstruction failed for $(method_name)")
            end
            oracles[(method_name, corruption)] = oracle
            push!(fit_rows, (; method=method_name, corruption, metrics...))
        end
    end

    trials = NamedTuple[]
    masses = unique(sort(config.prior_masses))
    for method_name in config.methods
        for online_updates in config.online_updates, trial in 1:config.trials
            seed = config.seed + 10_000 * online_updates + trial
            baseline_oracle = oracles[(method_name, first(config.corruptions))]
            baseline_policies = search_policy(
                game,
                baseline_oracle,
                method_name,
                online_updates,
                seed;
                prior_mass=0.0,
                epsilon=config.epsilon,
            )
            baseline_values = exact_policy_values(game, baseline_policies, initial)
            baseline_root_strategy = baseline_policies[initial]
            for corruption in config.corruptions
                push!(trials, (;
                    method=method_name,
                    corruption,
                    online_updates,
                    prior_mass=0.0,
                    trial,
                    seed,
                    lower_value=baseline_values.lower,
                    upper_value=baseline_values.upper,
                    exploitability=baseline_values.exploitability,
                    root_strategy=baseline_root_strategy,
                ))
            end
            for corruption in config.corruptions, prior_mass in masses
                oracle = oracles[(method_name, corruption)]
                policies = search_policy(
                    game,
                    oracle,
                    method_name,
                    online_updates,
                    seed;
                    prior_mass,
                    epsilon=config.epsilon,
                )
                values = exact_policy_values(game, policies, initial)
                root_strategy = policies[initial]
                push!(trials, (;
                    method=method_name,
                    corruption,
                    online_updates,
                    prior_mass,
                    trial,
                    seed,
                    lower_value=values.lower,
                    upper_value=values.upper,
                    exploitability=values.exploitability,
                    root_strategy,
                ))
            end
        end
    end

    summary_rows = summarize(trials)
    trial_header = vcat(
        [
            "method", "corruption", "online_updates", "prior_mass", "trial", "seed",
            "p1_value_vs_br_p2", "br_p1_value_vs_p2", "exploitability",
        ],
        ["root_p1_action_$(action)" for action in 1:3],
        ["root_p2_action_$(action)" for action in 1:3],
    )
    trial_data = [Any[
        row.method, row.corruption, row.online_updates, row.prior_mass, row.trial, row.seed,
        row.lower_value, row.upper_value, row.exploitability,
        row.root_strategy[1]..., row.root_strategy[2]...,
    ] for row in trials]
    summary_header = [
        "method", "corruption", "online_updates", "prior_mass", "trials",
        "exploitability_mean", "exploitability_stderr", "exploitability_p90",
        "paired_improvement_mean", "paired_improvement_stderr", "paired_win_rate",
    ]
    summary_data = [Any[getproperty(row, Symbol(name)) for name in summary_header]
                    for row in summary_rows]
    source_header = vcat(
        ["method", "step", "mode", "source_mass"],
        ["strategy_p1_action_$(action)" for action in 1:3],
        ["strategy_p2_action_$(action)" for action in 1:3],
        ["regret_p1_action_$(action)" for action in 1:3],
        ["regret_p2_action_$(action)" for action in 1:3],
    )
    fit_header = [
        "method", "corruption", "regret_relative_error", "strategy_relative_error",
        "regret_cosine", "strategy_tv",
    ]
    fit_data = [Any[getproperty(row, Symbol(name)) for name in fit_header] for row in fit_rows]

    write_csv(joinpath(config.output_dir, "trials.csv"), trial_header, trial_data)
    write_csv(joinpath(config.output_dir, "summary.csv"), summary_header, summary_data)
    write_csv(joinpath(config.output_dir, "source_priors.csv"), source_header, source_state_rows)
    write_csv(joinpath(config.output_dir, "fit_quality.csv"), fit_header, fit_data)
    print_results(source_rows, fit_rows, summary_rows)
    println("Wrote results to $(config.output_dir)")
    return nothing
end

main(ARGS)
