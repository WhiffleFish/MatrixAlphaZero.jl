@kwdef struct AlphaZeroSolver{S, OPT, RNG<:Random.AbstractRNG}
    max_steps       :: Int     = 10_000_000
    num_steps       :: Int     = 2048 * 8
    sim_depth       :: Int     = 50

    search          :: S

    update_epochs   :: Int     = 1
    num_batches     :: Int     = 4
    lr              :: Float32 = 3f-4
    lr_decay        :: Float32 = 1.0f0
    lr_min          :: Float32 = 0.0f0
    lr_max          :: Float32 = Float32(Inf)
    ema             :: Bool    = true
    ema_decay       :: Float32 = 0.99f0
    gae_lambda      :: Float64 = 0.95

    optimiser       :: OPT     = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(0.5f0),
        Flux.Optimisers.Adam(lr)
    )
    rng             :: RNG = Random.default_rng()
end

learning_rate(sol::AlphaZeroSolver, update::Integer) =
    clamp(sol.lr * sol.lr_decay ^ update, sol.lr_min, sol.lr_max)

function ema_update!(ema_model, model, decay::Real)
    for (ema_param, param) in zip(Flux.trainables(ema_model), Flux.trainables(model))
        @. ema_param = decay * ema_param + (1 - decay) * param
    end
    return ema_model
end

@kwdef mutable struct AlphaZeroPlanner{G<:MG, S} <: Policy
    game            :: G
    search          :: S
end

AlphaZeroPlanner(sol::AlphaZeroSolver, game::MG) =
    AlphaZeroPlanner(game, sol.search)

AlphaZeroPlanner(game::MG, sol::AlphaZeroSolver) =
    AlphaZeroPlanner(sol, game)

AlphaZeroPlanner(planner::AlphaZeroPlanner; search=planner.search) =
    AlphaZeroPlanner(planner.game, search)

oracle(search::Union{SMOOSSearch,MCTSSearch}) = search.oracle
oracle(planner::AlphaZeroPlanner) = oracle(planner.search)

function Flux.loadmodel!(planner::AlphaZeroPlanner, path::String)
    Flux.loadmodel!(oracle(planner), JLD2.load(path)["model_state"])
end

function Flux.loadmodel!(oracle::Union{FittedRegretModel,ActorCritic,CriticOnly}, path::String)
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

advance_transfer_tau(τ::Real, oos_iterations::Integer, transfer_weight::Real) =
    Float64(transfer_weight) * (Float64(τ) + Float64(oos_iterations))

struct LossScaledTransferState
    source_mass        :: Float64
    regret_confidence  :: Float64
    strategy_confidence:: Float64
end

function initial_search_state(search::Union{SMOOSSearch,MCTSSearch})
    uses_loss_scaled_transfer(search) || return search.τ
    return LossScaledTransferState(search.τ, search.regret_confidence, search.strategy_confidence)
end

search_callback_state(::SMOOSSearch, state::Real) = (; transfer_tau=state)
# Only surface transfer_tau when regret transfer is actually enabled, so plain
# (ActorCritic / no-transfer) MCTS runs keep their original callback shape.
search_callback_state(search::MCTSSearch, state::Real) =
    search.transfer_weight > 0 ? (; transfer_tau=state) : NamedTuple()

function search_callback_state(search::Union{SMOOSSearch,MCTSSearch}, state::LossScaledTransferState)
    active_search = with_search_state(search, search.oracle, state)
    regret_mass, strategy_mass = transfer_pseudo_masses(active_search)
    return (;
        transfer_source_mass=state.source_mass,
        transfer_regret_confidence=state.regret_confidence,
        transfer_strategy_confidence=state.strategy_confidence,
        transfer_regret_mass=regret_mass,
        transfer_strategy_mass=strategy_mass,
    )
end

with_search_state(search::SMOOSSearch, oracle, state::Real) =
    with_oracle(search, oracle; τ=state)
with_search_state(search::MCTSSearch, oracle, state::Real) =
    with_oracle(search, oracle; τ=state)

function with_search_state(search::Union{SMOOSSearch,MCTSSearch}, oracle, state::LossScaledTransferState)
    return with_oracle(search, oracle;
        τ=state.source_mass,
        regret_confidence=state.regret_confidence,
        strategy_confidence=state.strategy_confidence,
    )
end

advance_search_state(search::SMOOSSearch, state::Real) =
    advance_transfer_tau(state, search.oos_iterations, search.transfer_weight)
advance_search_state(search::MCTSSearch, state::Real) =
    advance_transfer_tau(state, search.tree_queries, search.transfer_weight)

advance_search_state(search::Union{SMOOSSearch,MCTSSearch}, state, train_stats) =
    advance_search_state(search, state)

function advance_search_state(
        search::Union{SMOOSSearch,MCTSSearch},
        state::LossScaledTransferState,
        train_stats,
    )
    config = search.loss_scaled_transfer::LossScaledTransfer
    raw_regret, raw_strategy = transfer_fit_confidence(train_stats, config)
    decay = config.confidence_ema_decay
    regret_confidence = decay * state.regret_confidence + (1 - decay) * raw_regret
    strategy_confidence = decay * state.strategy_confidence + (1 - decay) * raw_strategy
    return LossScaledTransferState(
        state.source_mass + search_budget(search),
        regret_confidence,
        strategy_confidence,
    )
end

run_selfplay(distributed::Bool, progress, game, search::SMOOSSearch, target_steps::Int, s0; ϵ, steps_done::Int, sim_depth::Int, gae_lambda) =
    distributed ?
        distributed_smoos(progress, game, search, target_steps, s0; ϵ, steps_done, sim_depth, gae_lambda) :
        serial_smoos(progress, game, search, target_steps, s0; ϵ, steps_done, sim_depth, gae_lambda)

run_selfplay(distributed::Bool, progress, game, search::MCTSSearch, target_steps::Int, s0; ϵ, steps_done::Int, sim_depth::Int, gae_lambda) =
    distributed ?
        distributed_mcts(progress, game, search, target_steps, s0; ϵ, steps_done, sim_depth, gae_lambda) :
        serial_mcts(progress, game, search, target_steps, s0; ϵ, steps_done, sim_depth, gae_lambda)

# RegretMatchingSearch backed by a FittedRegretModel emits regret/strategy
# targets via mcts_regret_sim; the ActorCritic path keeps policy targets.
mcts_selfplay_sim(search::MCTSSearch, game, s; ϵ, sim_depth, gae_lambda) =
    search.oracle isa FittedRegretModel ?
        mcts_regret_sim(search, game, s; ϵ, sim_depth, gae_lambda) :
        mcts_sim(search, game, s; ϵ, sim_depth)

function MarkovGames.behavior_info(policy::AlphaZeroPlanner, s)
    return search_behavior_info(policy.search, policy.game, s)
end

function search_behavior_info(search::SMOOSSearch, game::MG, s)
    A1, A2 = actions(game)
    if isterminal(game, s)
        x, y = uniform(length(A1)), uniform(length(A2))
        return ProductDistribution(SparseCat(A1, x), SparseCat(A2, y)), (; v=0.0)
    end
    (yr, ys), info = fitted_smoos_info(search, game, s; ϵ=search.ϵ(1))
    x = normalized_or_uniform(ys[1])
    y = normalized_or_uniform(ys[2])
    v = oracle_state_value(search.oracle, game, s)
    return ProductDistribution(SparseCat(A1, x), SparseCat(A2, y)), (; info..., regret=yr, strategy=ys, v)
end

function search_behavior_info(search::MCTSSearch, game::MG, s)
    A1, A2 = actions(game)
    (x, y, v), info = search_info(search, game, s; ϵ=search.ϵ(1))
    return ProductDistribution(SparseCat(A1, x), SparseCat(A2, y)), (; info..., policy=(x, y), v)
end

MarkovGames.behavior(policy::AlphaZeroPlanner, s) = first(behavior_info(policy, s))

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG; s0=initialstate(game), cb=(info)->())
    sol.max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    sol.num_steps > 0 || throw(ArgumentError("num_steps must be positive"))
    sol.sim_depth > 0 || throw(ArgumentError("sim_depth must be positive"))
    sol.lr > 0 || throw(ArgumentError("lr must be positive"))
    sol.lr_decay > 0 || throw(ArgumentError("lr_decay must be positive"))
    0 ≤ sol.lr_min ≤ sol.lr_max || throw(ArgumentError("lr bounds must satisfy 0 <= lr_min <= lr_max"))
    0 ≤ sol.gae_lambda ≤ 1 || throw(ArgumentError("gae_lambda must be in [0, 1]"))
    distributed = Distributed.nprocs() > 1
    online_oracle = oracle(sol.search)
    if uses_loss_scaled_transfer(sol.search) && !(online_oracle isa FittedRegretModel)
        throw(ArgumentError("LossScaledTransfer requires a FittedRegretModel oracle"))
    end
    ema_oracle = deepcopy(online_oracle)
    progress = Progress(sol.max_steps, safe_lock=false)
    opt_state = Flux.setup(sol.optimiser, online_oracle)
    cb_oracle = sol.ema ? ema_oracle : online_oracle
    prev_cb_oracle = deepcopy(cb_oracle)
    steps_done = 0
    update = 0
    search_state = initial_search_state(sol.search)
    active_learning_rate = learning_rate(sol, update)
    active_learning_rate == sol.lr || Flux.Optimisers.adjust!(opt_state; eta=active_learning_rate)
    call(cb, merge((;
        oracle=cb_oracle,
        iter=0,
        update=0,
        steps_done,
        max_steps=sol.max_steps,
        sim_depth=sol.sim_depth,
        learning_rate=active_learning_rate,
        exploration_epsilon=sol.search.ϵ(1),
        online_oracle,
        ema_oracle=sol.ema ? ema_oracle : nothing,
    ), search_callback_state(sol.search, search_state)))
    while steps_done < sol.max_steps
        update += 1
        target_steps = min(sol.num_steps, sol.max_steps - steps_done)
        ϵ = sol.search.ϵ(update)
        selfplay_oracle = sol.ema ? ema_oracle : online_oracle
        iter_search = with_search_state(sol.search, selfplay_oracle, search_state)
        hists = run_selfplay(distributed, progress, game, iter_search, target_steps, s0; ϵ, steps_done, sim_depth=sol.sim_depth, gae_lambda=sol.gae_lambda)
        batch = merge_histories(hists)
        samples_added = length(batch.v)
        iszero(samples_added) && break
        steps_done += samples_added
        train_stats = train!(sol, online_oracle, batch; opt_state)
        sol.ema && ema_update!(ema_oracle, online_oracle, sol.ema_decay)
        search_state = advance_search_state(sol.search, search_state, train_stats)
        next_learning_rate = learning_rate(sol, update)
        next_learning_rate == active_learning_rate || Flux.Optimisers.adjust!(opt_state; eta=next_learning_rate)
        cb_oracle = sol.ema ? ema_oracle : online_oracle
        cb_info = merge(
            merge((oracle=cb_oracle, iter=update, update, steps_done, max_steps=sol.max_steps, sim_depth=sol.sim_depth, samples_added, learning_rate=active_learning_rate, exploration_epsilon=ϵ, online_oracle, ema_oracle=sol.ema ? ema_oracle : nothing), search_callback_state(sol.search, search_state)),
            selfplay_metrics(hists),
            training_metrics(train_stats),
            (; minibatch_metrics=training_minibatch_metrics(train_stats)),
            batch_metrics(batch),
            oracle_metrics(cb_oracle, prev_cb_oracle, batch),
        )
        call(cb, cb_info)
        prev_cb_oracle = deepcopy(cb_oracle)
        active_learning_rate = next_learning_rate
    end
    finish!(progress)
    final_oracle = sol.ema ? ema_oracle : online_oracle
    return AlphaZeroPlanner(game, with_search_state(sol.search, final_oracle, search_state))
end

function distributed_mcts(progress, game, search, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, sim_depth::Int=search.max_depth, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    n_tasks = max(1, Distributed.nworkers())
    while collected < num_steps
        task_count = min(n_tasks, num_steps - collected)
        new_hists = pmap(1:task_count) do _
            mcts_selfplay_sim(search, game, rand(s0); ϵ, sim_depth, gae_lambda)
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

function serial_mcts(progress, game, search, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, sim_depth::Int=search.max_depth, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    while collected < num_steps
        hist = trim_history(mcts_selfplay_sim(search, game, rand(s0); ϵ, sim_depth, gae_lambda), num_steps - collected)
        n = length(hist.v)
        iszero(n) && break
        push!(hists, hist)
        collected += n
        update!(progress, steps_done + collected)
    end
    return hists
end

function distributed_smoos(progress, game, search, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, sim_depth::Int=search.max_depth, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    n_tasks = max(1, Distributed.nworkers())
    while collected < num_steps
        task_count = min(n_tasks, num_steps - collected)
        new_hists = pmap(1:task_count) do _
            smoos_sim(search, game, rand(s0); ϵ, sim_depth, gae_lambda)
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

function serial_smoos(progress, game, search, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0, sim_depth::Int=search.max_depth, gae_lambda=0.95)
    hists = NamedTuple[]
    collected = 0
    while collected < num_steps
        hist = trim_history(smoos_sim(search, game, rand(s0); ϵ, sim_depth, gae_lambda), num_steps - collected)
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
    common = (;
        s = hist.s[1:n],
        r = hist.r[1:n],
        v = hist.v[1:n],
        search_time = hist.search_time[1:n],
    )
    if hasproperty(hist, :regret)
        return merge(common, (;
            regret = (hist.regret[1][1:n], hist.regret[2][1:n]),
            strategy = (hist.strategy[1][1:n], hist.strategy[2][1:n]),
        ))
    else
        return merge(common, (;
            policy = (hist.policy[1][1:n], hist.policy[2][1:n]),
        ))
    end
end

function merge_histories(hists)
    s = Vector{Float32}[]
    r = Float64[]
    v = Float64[]
    p1 = Vector{Float64}[]
    p2 = Vector{Float64}[]
    if isempty(hists) || !hasproperty(first(hists), :regret)
        for hist in hists
            append!(s, hist.s)
            append!(r, hist.r)
            append!(v, hist.v)
            append!(p1, hist.policy[1])
            append!(p2, hist.policy[2])
        end
        return (; s, r, v, policy=(p1, p2))
    end

    r1 = Vector{Float64}[]
    r2 = Vector{Float64}[]
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
