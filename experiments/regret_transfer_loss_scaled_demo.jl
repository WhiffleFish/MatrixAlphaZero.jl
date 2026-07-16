include("regret_transfer_toy.jl")

# A focused demonstration of the production LossScaledTransfer rule. Both
# conditions use the same exact value oracle and the same RM+ search seeds; the
# transferred condition additionally receives imperfect fitted regret/strategy
# tables from a long source search.

const DEMO_HORIZON = 3
const DEMO_SOURCE_UPDATES = 4096
const DEMO_ONLINE_UPDATES = (4, 8, 16, 32, 64, 128, 256, 500)
const DEMO_FIT_CONDITIONS = (
    (corruption=0.25, confidence=0.75),
    (corruption=0.50, confidence=0.50),
    (corruption=0.50, confidence=0.25),
    (corruption=0.75, confidence=0.25),
)
const DEMO_TRIALS = 100
const DEMO_SEED = 20260714
const DEMO_METHOD = "plus"
const DEMO_OUTPUT = joinpath(@__DIR__, "regret_transfer_loss_scaled_results", "summary.csv")

function loss_scaled_search_policy(
        game,
        oracle,
        method_name,
        updates,
        seed;
        source_mass,
        confidence,
        transfer,
    )
    policies = Dict{SequentialState,NTuple{2,Vector{Float64}}}()
    for state in nonterminal_states(game)
        state_seed = seed + 100 * state.step + state.mode
        search = AZ.MCTSSearch(;
            oracle,
            tree_queries=updates + 1,
            max_depth=game.horizon - state.step + 1,
            search_style=AZ.RegretMatchingSearch(; method=method_object(method_name)),
            τ=Float64(source_mass),
            transfer_weight=0.0,
            loss_scaled_transfer=transfer,
            regret_confidence=confidence,
            strategy_confidence=confidence,
            ϵ=_ -> 0.0,
        )
        Random.seed!(state_seed)
        (strategy1, strategy2, _), _ = AZ.search_info(search, game, state; ϵ=0.0)
        policies[state] = (Float64.(strategy1), Float64.(strategy2))
    end
    return policies
end

function demo_summary(values)
    gaps = getproperty.(values, :transfer_exploitability)
    baselines = getproperty.(values, :baseline_exploitability)
    improvements = baselines .- gaps
    return (;
        baseline_mean=mean(baselines),
        baseline_stderr=stderr(baselines),
        transfer_mean=mean(gaps),
        transfer_stderr=stderr(gaps),
        paired_improvement=mean(improvements),
        paired_improvement_stderr=stderr(improvements),
        paired_win_rate=mean(>(0), improvements),
    )
end

function main_demo()
    game = SequentialRPS(DEMO_HORIZON)
    initial = rand(initialstate(game))
    equilibrium = equilibrium_tables(game)
    value_oracle = empty_oracle(game, equilibrium)
    source = make_source(
        game,
        equilibrium,
        DEMO_METHOD,
        DEMO_SOURCE_UPDATES,
        DEMO_SEED - 1,
    )
    transfer = AZ.LossScaledTransfer()
    rows = NamedTuple[]

    println("Loss-scaled fitted regret transfer demonstration")
    @printf(
        "Game: horizon=%d, states=%d, method=RM+, source updates=%d, trials=%d\n",
        DEMO_HORIZON,
        length(nonterminal_states(game)),
        DEMO_SOURCE_UPDATES,
        DEMO_TRIALS,
    )
    @printf(
        "Transfer: regret_scale=%.2f strategy_scale=%.2f reach_power=%.2f\n\n",
        transfer.regret_scale,
        transfer.strategy_scale,
        transfer.reach_power,
    )

    for condition in DEMO_FIT_CONDITIONS
        corruption = condition.corruption
        confidence = condition.confidence
        fitted_oracle = corrupted_source_oracle(
            equilibrium,
            source,
            corruption,
            DEMO_SEED + 100_000,
        )
        fit = reconstruction_metrics(game, DEMO_METHOD, source, fitted_oracle)
        @printf(
            "Fit corruption %.2f: R-relative=%.3f R-cosine=%.3f S-TV=%.3f confidence=%.2f\n",
            corruption,
            fit.regret_relative_error,
            fit.regret_cosine,
            fit.strategy_tv,
            confidence,
        )

        for updates in DEMO_ONLINE_UPDATES
            trial_values = NamedTuple[]
            for trial in 1:DEMO_TRIALS
                seed = DEMO_SEED + 10_000 * updates + trial
                baseline_policy = search_policy(
                    game,
                    value_oracle,
                    DEMO_METHOD,
                    updates,
                    seed;
                    prior_mass=0.0,
                    epsilon=0.0,
                )
                transfer_policy = loss_scaled_search_policy(
                    game,
                    fitted_oracle,
                    DEMO_METHOD,
                    updates,
                    seed;
                    source_mass=DEMO_SOURCE_UPDATES,
                    confidence,
                    transfer,
                )
                baseline = exact_policy_values(game, baseline_policy, initial)
                transferred = exact_policy_values(game, transfer_policy, initial)
                push!(trial_values, (;
                    baseline_exploitability=baseline.exploitability,
                    transfer_exploitability=transferred.exploitability,
                ))
            end

            summary = demo_summary(trial_values)
            regret_mass, strategy_mass = AZ.transfer_pseudo_masses(
                AZ.MCTSSearch(;
                    oracle=fitted_oracle,
                    tree_queries=updates + 1,
                    τ=Float64(DEMO_SOURCE_UPDATES),
                    loss_scaled_transfer=transfer,
                    regret_confidence=confidence,
                    strategy_confidence=confidence,
                ),
            )
            row = (;
                method=DEMO_METHOD,
                corruption,
                confidence,
                online_updates=updates,
                tree_queries=updates + 1,
                source_mass=DEMO_SOURCE_UPDATES,
                root_regret_mass=regret_mass,
                root_strategy_mass=strategy_mass,
                fit_regret_relative_error=fit.regret_relative_error,
                fit_regret_cosine=fit.regret_cosine,
                fit_strategy_tv=fit.strategy_tv,
                summary...,
            )
            push!(rows, row)
            @printf(
                "  updates=%2d queries=%2d mR=%5.2f mS=%5.2f | value-only %.4f +/- %.4f | transfer %.4f +/- %.4f | gain %.4f +/- %.4f | wins %.0f%%\n",
                updates,
                updates + 1,
                regret_mass,
                strategy_mass,
                summary.baseline_mean,
                summary.baseline_stderr,
                summary.transfer_mean,
                summary.transfer_stderr,
                summary.paired_improvement,
                summary.paired_improvement_stderr,
                100 * summary.paired_win_rate,
            )
        end
        println()
    end

    header = String.(propertynames(first(rows)))
    data = [Any[getproperty(row, Symbol(name)) for name in header] for row in rows]
    write_csv(DEMO_OUTPUT, header, data)
    println("Wrote $(DEMO_OUTPUT)")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_demo()
