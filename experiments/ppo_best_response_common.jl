# Option handling shared by the per-domain `ppo_best_response.jl` scripts. Each script
# builds the three round-robin solvers, then calls
# `ExperimentTools.run_ppo_best_response_suite` with the settings parsed here.

using Pkg
using LinearAlgebra
using ExperimentTools
using MarkovGames

const BR_SOLVER_NAMES = ("zero_oracle", "value_oracle", "full_solver")

# Paper round-robin search settings first, then PPO and evaluation settings.
const BR_DEFAULT_OPTIONS = Dict(
    "solvers" => join(BR_SOLVER_NAMES, ","),
    "players" => "both",
    "checkpoint" => "latest",
    "tree-queries" => "50",
    "max-depth" => "5",
    "search-epsilon" => "0.1",
    "prior-scale" => "2.5",
    "max-steps" => "50",
    "ppo-inits" => "bc",
    "total-timesteps" => "500000",
    "num-steps" => "128",
    "num-envs" => "8",
    "num-minibatches" => "4",
    "update-epochs" => "4",
    "lr" => "2.5e-4",
    "gamma" => "game",
    "gae-lambda" => "0.95",
    "clip-coef" => "0.2",
    "ent-coeff" => "0.01",
    "v-coef" => "0.5",
    "target-kl" => "0.02",
    "bc-episodes" => "64",
    "bc-epochs" => "100",
    "select-runs" => "200",
    "eval-runs" => "1000",
    "fail-fast" => "false",
)

function parse_br_options(args, defaults::Dict{String,String})
    opts = merge(BR_DEFAULT_OPTIONS, defaults)
    opts["test"] = "false"
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        name = key[3:end]
        haskey(opts, name) || error("Unknown option $(key)")
        if name == "test"
            opts[name] = "true"
            i += 1
            continue
        end
        i < length(args) || error("Missing value after $(key)")
        opts[name] = args[i + 1]
        i += 2
    end
    return opts
end

split_option(s::AbstractString) = String.(filter(!isempty, strip.(split(s, ","))))

function parse_br_players(spec::AbstractString)
    spec == "both" && return [1, 2]
    return map(split_option(spec)) do item
        item in ("1", "p1") && return 1
        item in ("2", "p2") && return 2
        error("Unknown player $(item)")
    end
end

"""
    br_settings(opts, game)

Turn parsed options into search settings and the keyword arguments for
`run_ppo_best_response_suite`. `--test` shrinks every budget to a smoke test.
"""
function br_settings(opts, game)
    test = parse(Bool, opts["test"])
    int(key) = parse(Int, opts[key])
    f32(key) = parse(Float32, opts[key])
    gamma = opts["gamma"] == "game" ? Float32(discount(game)) : f32("gamma")
    target_kl = opts["target-kl"] == "none" ? nothing : f32("target-kl")
    solvers = split_option(opts["solvers"])
    unknown = setdiff(solvers, BR_SOLVER_NAMES)
    isempty(unknown) || error("Unknown solvers $(unknown)")
    ppo_inits = split_option(opts["ppo-inits"])
    all(in(("bc", "scratch")), ppo_inits) || error("--ppo-inits takes bc and/or scratch")

    search = (;
        queries=int("tree-queries"),
        max_depth=int("max-depth"),
        search_epsilon=parse(Float64, opts["search-epsilon"]),
        prior_scale=parse(Float64, opts["prior-scale"]),
    )
    ppo = (;
        total_timesteps=int("total-timesteps"),
        num_steps=int("num-steps"),
        num_envs=int("num-envs"),
        num_minibatches=int("num-minibatches"),
        update_epochs=int("update-epochs"),
    )
    bc = (; episodes=int("bc-episodes"), epochs=int("bc-epochs"))
    runs = (; select_runs=int("select-runs"), eval_runs=int("eval-runs"))
    max_steps = int("max-steps")
    if test
        search = merge(search, (; queries=2, max_depth=2,
            prior_scale=min(search.prior_scale, 2.0)))
        ppo = (; total_timesteps=64, num_steps=8, num_envs=2, num_minibatches=2,
            update_epochs=1)
        bc = (; episodes=2, epochs=2)
        runs = (; select_runs=2, eval_runs=2)
        max_steps = min(max_steps, 5)
    end

    ppo_config = PPOBestResponseConfig(;
        ppo...,
        lr=f32("lr"),
        gamma,
        gae_lambda=f32("gae-lambda"),
        clip_coef=f32("clip-coef"),
        ent_coeff=f32("ent-coeff"),
        v_coef=f32("v-coef"),
        target_kl,
        max_steps,
    )
    suite = (;
        solvers,
        players=parse_br_players(opts["players"]),
        seed=int("seed"),
        fail_fast=test || parse(Bool, opts["fail-fast"]),
        ppo_config,
        ppo_inits,
        bc_config=BehaviorCloneConfig(; bc...),
        runs...,
        max_steps,
    )
    return (; test, search, suite)
end

function cleanrl_source()
    info = Pkg.dependencies()[Base.PkgId(ExperimentTools.CleanRL).uuid]
    return something(info.tree_hash, info.source)
end

function br_metadata(settings)
    (; search, suite) = settings
    return (;
        tree_queries=search.queries,
        max_depth=search.max_depth,
        search_epsilon=search.search_epsilon,
        prior_scale=search.prior_scale,
        ppo_inits=join(suite.ppo_inits, ";"),
        total_timesteps=suite.ppo_config.total_timesteps,
        num_envs=suite.ppo_config.num_envs,
        gamma=suite.ppo_config.gamma,
        target_kl=something(suite.ppo_config.target_kl, NaN),
        bc_episodes=suite.bc_config.episodes,
        max_steps=suite.max_steps,
        threads=Threads.nthreads(),
        cleanrl_source=cleanrl_source(),
    )
end

# Planners and PPO environments run on Julia threads; keep BLAS from oversubscribing.
limit_blas_threads() = Threads.nthreads() > 1 && BLAS.set_num_threads(1)
