@kwdef struct AlphaZeroSolver{OPT, RNG<:Random.AbstractRNG, O, SP}
    max_steps       :: Int     = 10_000_000
    num_steps       :: Int     = 2048 * 8

    oracle          :: O
    smoos_params    :: SP      = SMOOSParams(;oracle)

    update_epochs   :: Int     = 1
    num_batches     :: Int     = 4
    lr              :: Float32 = 3f-4
    lr_decay        :: Float32 = 1.0f0
    ema_decay       :: Float32 = 0.99f0
    gae_lambda      :: Float64 = 0.95
    transfer_weight :: Float64 = 0.1

    optimiser       :: OPT     = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(0.5f0),
        Flux.Optimisers.Adam(lr)
    )
    rng             :: RNG = Random.default_rng()
end

function ema_update!(ema_model, model, decay::Real)
    for (ema_param, param) in zip(Flux.trainables(ema_model), Flux.trainables(model))
        @. ema_param = decay * ema_param + (1 - decay) * param
    end
    return ema_model
end

@kwdef mutable struct AlphaZeroPlanner{G<:MG, Oracle, SP} <: Policy
    game            :: G
    oracle          :: Oracle
    smoos_params    :: SP
end

function AlphaZeroPlanner(
        game::MG,
        oracle;
        oos_iterations  = 0,
        τ               = 0.0,
        max_depth       = typemax(Int),
        ϵ               = t -> 0.0,
        smoos_params    = SMOOSParams(; oracle, oos_iterations, τ, max_depth, ϵ),
    )
    return AlphaZeroPlanner(; game, oracle, smoos_params=with_oracle(smoos_params, oracle))
end

function Flux.loadmodel!(planner::AlphaZeroPlanner, path::String)
    Flux.loadmodel!(planner.oracle, JLD2.load(path)["model_state"])
end

function Flux.loadmodel!(oracle::FittedRegretModel, path::String)
    Flux.loadmodel!(oracle, JLD2.load(path)["model_state"])
end

function load_oracle(path)
    if isdir(path)
        return jldopen(joinpath(path, "oracle.jld2"))["oracle"]
    elseif isfile(path)
        return jldopen(path)["oracle"]
    else
        error("$(path) is neither file nor dir")
    end
end

AlphaZeroPlanner(sol::AlphaZeroSolver, game::MG; kwargs...) =
    AlphaZeroPlanner(game, sol.smoos_params.oracle; smoos_params=sol.smoos_params, kwargs...)

AlphaZeroPlanner(game::MG, sol::AlphaZeroSolver; kwargs...) =
    AlphaZeroPlanner(sol, game; kwargs...)

AlphaZeroPlanner(planner::AlphaZeroPlanner; kwargs...) =
    AlphaZeroPlanner(planner.game, planner.oracle; smoos_params=planner.smoos_params, kwargs...)

SMOOSParams(planner::AlphaZeroPlanner; kwargs...) =
    SMOOSParams(; oracle=planner.oracle,
        oos_iterations=planner.smoos_params.oos_iterations,
        τ=planner.smoos_params.τ,
        max_depth=planner.smoos_params.max_depth,
        ϵ=planner.smoos_params.ϵ,
        kwargs...
    )

advance_transfer_tau(τ::Real, oos_iterations::Integer, transfer_weight::Real) =
    Float64(transfer_weight) * (Float64(τ) + Float64(oos_iterations))

function MarkovGames.behavior_info(policy::AlphaZeroPlanner, s)
    (; oracle, game, smoos_params) = policy
    A1, A2 = actions(game)
    if isterminal(game, s)
        x, y = uniform(length(A1)), uniform(length(A2))
        return ProductDistribution(SparseCat(A1, x), SparseCat(A2, y)), (; v=0.0)
    end
    (yr, ys), info = fitted_smoos_info(with_oracle(smoos_params, oracle), game, s; ϵ=smoos_params.ϵ(1))
    x = normalized_or_uniform(ys[1])
    y = normalized_or_uniform(ys[2])
    v = oracle_state_value(oracle, game, s)
    return ProductDistribution(SparseCat(A1, x), SparseCat(A2, y)), (; info..., regret=yr, strategy=ys, v)
end

MarkovGames.behavior(policy::AlphaZeroPlanner, s) = first(behavior_info(policy, s))

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG; s0=initialstate(game), cb=(info)->())
    sol.max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    sol.num_steps > 0 || throw(ArgumentError("num_steps must be positive"))
    0 <= sol.gae_lambda <= 1 || throw(ArgumentError("gae_lambda must be in [0, 1]"))
    distributed = Distributed.nprocs() > 1
    online_oracle = sol.smoos_params.oracle
    ema_oracle = deepcopy(online_oracle)
    progress = Progress(sol.max_steps, safe_lock=false)
    opt_state = Flux.setup(sol.optimiser, online_oracle)
    prev_ema_oracle = deepcopy(ema_oracle)
    cb_oracle = ema_oracle
    steps_done = 0
    update = 0
    transfer_tau = sol.smoos_params.τ
    call(cb, (;
        oracle=cb_oracle,
        iter=0,
        update=0,
        steps_done,
        max_steps=sol.max_steps,
        exploration_epsilon=sol.smoos_params.ϵ(1),
        transfer_tau,
        online_oracle,
        ema_oracle,
    ))
    while steps_done < sol.max_steps
        update += 1
        target_steps = min(sol.num_steps, sol.max_steps - steps_done)
        ϵ = sol.smoos_params.ϵ(update)
        selfplay_oracle = ema_oracle
        smoos_params = with_oracle(sol.smoos_params, selfplay_oracle; τ=transfer_tau)
        hists = if distributed
            distributed_smoos(progress, game, smoos_params, target_steps, s0; ϵ, steps_done, gae_lambda=sol.gae_lambda)
        else
            serial_smoos(progress, game, smoos_params, target_steps, s0; ϵ, steps_done, gae_lambda=sol.gae_lambda)
        end
        batch = merge_histories(hists)
        samples_added = length(batch.v)
        iszero(samples_added) && break
        steps_done += samples_added
        train_stats = train!(sol, online_oracle, batch; opt_state)
        ema_update!(ema_oracle, online_oracle, sol.ema_decay)
        transfer_tau = advance_transfer_tau(transfer_tau, sol.smoos_params.oos_iterations, sol.transfer_weight)
        if sol.lr_decay < 1f0
            Flux.Optimisers.adjust!(opt_state; eta = sol.lr * sol.lr_decay ^ update)
        end
        cb_oracle = ema_oracle
        cb_info = merge(
            (oracle=cb_oracle, iter=update, update, steps_done, max_steps=sol.max_steps, samples_added, exploration_epsilon=ϵ, transfer_tau, online_oracle, ema_oracle),
            selfplay_metrics(hists),
            training_metrics(train_stats),
            (; minibatch_metrics=training_minibatch_metrics(train_stats)),
            batch_metrics(batch),
            oracle_metrics(ema_oracle, prev_ema_oracle, batch),
        )
        call(cb, cb_info)
        prev_ema_oracle = deepcopy(ema_oracle)
    end
    finish!(progress)
    return AlphaZeroPlanner(game, ema_oracle; smoos_params=with_oracle(sol.smoos_params, ema_oracle; τ=transfer_tau))
end

function lambda_gae_targets(rewards, values, bootstrap, γ, λ)
    targets = Vector{Float64}(undef, length(rewards))
    adv = 0.0
    vnext = Float64(bootstrap)
    for t ∈ reverse(eachindex(rewards))
        δ = Float64(rewards[t]) + γ * vnext - Float64(values[t])
        adv = δ + γ * λ * adv
        targets[t] = adv + Float64(values[t])
        vnext = Float64(values[t])
    end
    return targets
end

function smoos_sim(params::SMOOSParams, game::MG, s; progress=false, ϵ=0.30, gae_lambda=0.95)
    d = params.max_depth
    A1, A2 = actions(game)
    γ = discount(game)
    t = 1
    rewards = Float64[]
    values = Float64[]
    search_time_hist = Float64[]
    s_hist = Vector{Float32}[]
    regret_hist = (Vector{Float64}[], Vector{Float64}[])
    strategy_hist = (Vector{Float64}[], Vector{Float64}[])
    p = Progress(d, enabled=progress)

    while (t < d) && !isterminal(game, s)
        search_start = time()
        (yr, ys), _info = fitted_smoos_info(params, game, s; ϵ)
        search_time = time() - search_start

        x = normalized_or_uniform(ys[1])
        y = normalized_or_uniform(ys[2])
        a_idxs = Tuple(action_idx_from_probs(x, y))
        a = (A1[a_idxs[1]], A2[a_idxs[2]])
        sp, r = @gen(:sp, :r)(game, s, a)
        r = zs_reward_scalar(r)
        push!(search_time_hist, search_time)
        push!(s_hist, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(values, oracle_state_value(params.oracle, game, s))
        push!(rewards, Float64(r))
        push!(regret_hist[1], Float64.(yr[1]))
        push!(regret_hist[2], Float64.(yr[2]))
        push!(strategy_hist[1], Float64.(ys[1]))
        push!(strategy_hist[2], Float64.(ys[2]))
        t += 1
        s = sp
        next!(p)
    end
    bootstrap = isterminal(game, s) ? 0.0 : oracle_state_value(params.oracle, game, s)
    v_hist = lambda_gae_targets(rewards, values, bootstrap, γ, gae_lambda)
    finish!(p)
    return (;
        s = s_hist,
        r = rewards,
        v = v_hist,
        search_time = search_time_hist,
        regret = regret_hist,
        strategy = strategy_hist,
    )
end

function distributed_smoos(progress, game, params, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    n_tasks = max(1, Distributed.nworkers())
    while collected < num_steps
        task_count = min(n_tasks, num_steps - collected)
        new_hists = pmap(1:task_count) do _
            smoos_sim(params, game, rand(s0); ϵ, gae_lambda)
        end
        made_progress = false
        for hist in new_hists
            remaining = num_steps - collected
            iszero(remaining) && break
            hist = trim_history(hist, remaining)
            n = length(hist.v)
            iszero(n) && continue
            push!(hists, hist)
            collected += n
            made_progress = true
            update!(progress, steps_done + collected)
        end
        made_progress || break
    end
    return hists
end

function serial_smoos(progress, game, params, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    while collected < num_steps
        hist = trim_history(smoos_sim(params, game, rand(s0); ϵ, gae_lambda), num_steps - collected)
        n = length(hist.v)
        iszero(n) && break
        push!(hists, hist)
        collected += n
        update!(progress, steps_done + collected)
    end
    return hists
end

function trim_history(hist::NamedTuple, max_steps::Int)
    n = min(max_steps, length(hist.v))
    return (;
        s = hist.s[1:n],
        r = hist.r[1:n],
        v = hist.v[1:n],
        search_time = hist.search_time[1:n],
        regret = (hist.regret[1][1:n], hist.regret[2][1:n]),
        strategy = (hist.strategy[1][1:n], hist.strategy[2][1:n]),
    )
end

function merge_histories(hists)
    s = Vector{Float32}[]
    r = Float64[]
    v = Float64[]
    r1 = Vector{Float64}[]
    r2 = Vector{Float64}[]
    p1 = Vector{Float64}[]
    p2 = Vector{Float64}[]
    for hist in hists
        append!(s, hist.s)
        append!(r, hist.r)
        append!(v, hist.v)
        append!(r1, hist.regret[1])
        append!(r2, hist.regret[2])
        append!(p1, hist.strategy[1])
        append!(p2, hist.strategy[2])
    end
    return (; s, r, v, regret=(r1, r2), strategy=(p1, p2))
end
