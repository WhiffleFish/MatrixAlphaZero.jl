@kwdef struct AlphaZeroSolver{OPT, RNG<:Random.AbstractRNG, O, MP}
    max_steps       ::  Int     = 10_000_000
    num_steps       ::  Int     = 2048 * 8

    # MCTS
    oracle          ::  O
    mcts_params     ::  MP      = MCTSParams(;oracle)

    # Training args
    update_epochs   ::  Int     = 1
    num_batches     ::  Int     = 4
    lr              ::  Float32 = 3f-4
    lr_decay        ::  Float32 = 1.0f0  # multiplicative LR factor applied each iteration (1 = no decay)
    ema_decay       ::  Float32 = 0.99f0

    optimiser       ::  OPT     = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(0.5f0),
        Flux.Optimisers.Adam(lr)
    )
    rng             ::  RNG = Random.default_rng()
end

function ema_update!(ema_model, model, decay::Real)
    for (ema_param, param) in zip(Flux.trainables(ema_model), Flux.trainables(model))
        @. ema_param = decay * ema_param + (1 - decay) * param
    end
    return ema_model
end


@kwdef mutable struct AlphaZeroPlanner{G<:MG, Oracle} <: Policy
    game            ::  G
    oracle          ::  Oracle
    search_style    ::  RegretMatchingSearch
    max_iter        ::  Int
    max_time        ::  Float64
    max_depth       ::  Int
end

function AlphaZeroPlanner(
        game::MG,
        oracle;
        max_iter        =   0,
        max_time        =   Inf,
        max_depth       =   typemax(Int),
        search_style    = RegretMatchingSearch()
    )
    return AlphaZeroPlanner(;
        game,
        oracle,
        search_style,
        max_iter,
        max_time,
        max_depth,
    )
end

function Flux.loadmodel!(planner::AlphaZeroPlanner, path::String)
    Flux.loadmodel!(planner.oracle, JLD2.load(path)["model_state"])
end

function Flux.loadmodel!(oracle::ActorCritic, path::String)
    Flux.loadmodel!(oracle, JLD2.load(path)["model_state"])
end

function load_oracle(path)
    if isdir(path)
        return jldopen(joinpath(path, "oracle.jld2"))["oracle"]
    elseif isfile(path)
        return jldopen(path)["oracle"]
    else # when would this ever be hit? lmao
        error("$(path) is neither file nor dir")
    end
end

AlphaZeroPlanner(sol::AlphaZeroSolver, game::MG; kwargs...) = AlphaZeroPlanner(
    game, sol.mcts_params.oracle;
    max_iter        = sol.mcts_params.tree_queries,
    max_time        = sol.mcts_params.max_time,
    max_depth       = sol.mcts_params.max_depth,
    search_style    = sol.mcts_params.search_style,
    kwargs...
)

AlphaZeroPlanner(game::MG, sol::AlphaZeroSolver; kwargs...) = AlphaZeroPlanner(sol, game; kwargs...)

AlphaZeroPlanner(planner::AlphaZeroPlanner; kwargs...) = AlphaZeroPlanner(
    planner.game,
    planner.oracle;
    max_iter        =   planner.max_iter,
    max_time        =   planner.max_time,
    max_depth       =   planner.max_depth,
    search_style    =   planner.search_style,
    kwargs...
)

function MCTSParams(planner::AlphaZeroPlanner; kwargs...)
    return MCTSParams(;
        tree_queries = planner.max_iter,
        max_depth = planner.max_depth,
        max_time = planner.max_time,
        search_style = planner.search_style,
        oracle = planner.oracle,
        kwargs...
    )
end

function MarkovGames.behavior_info(policy::AlphaZeroPlanner, s)
    (;oracle, game, max_iter, max_time, max_depth, search_style) = policy
    A1, A2 = actions(game)
    (x,y,v), info = search_info(MCTSParams(;oracle, tree_queries=max_iter, max_depth, max_time, search_style), game, s)
    return ProductDistribution(SparseCat(A1,x), SparseCat(A2,y)), (;info..., v)
end

MarkovGames.behavior(policy::AlphaZeroPlanner, s) = first(behavior_info(policy, s))

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG; s0=initialstate(game), cb=(info)->())
    sol.max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    sol.num_steps > 0 || throw(ArgumentError("num_steps must be positive"))
    distributed = Distributed.nprocs() > 1
    online_oracle = sol.mcts_params.oracle
    ema_oracle = deepcopy(online_oracle)
    progress = Progress(sol.max_steps, safe_lock=false)
    opt_state = Flux.setup(sol.optimiser, online_oracle)
    prev_ema_oracle = deepcopy(ema_oracle)
    cb_oracle = ema_oracle
    steps_done = 0
    update = 0
    call(cb, (;oracle=cb_oracle, iter=0, update=0, steps_done, max_steps=sol.max_steps, online_oracle, ema_oracle))
    while steps_done < sol.max_steps
        update += 1
        target_steps = min(sol.num_steps, sol.max_steps - steps_done)
        ϵ = sol.mcts_params.ϵ(update)
        selfplay_oracle = ema_oracle
        mcts_params = with_oracle(sol.mcts_params, selfplay_oracle)
        hists = if distributed
            distributed_mcts(progress, game, mcts_params, target_steps, s0; ϵ, steps_done)
        else
            serial_mcts(progress, game, mcts_params, target_steps, s0; ϵ, steps_done)
        end
        batch = merge_histories(hists)
        samples_added = length(batch.v)
        iszero(samples_added) && break
        steps_done += samples_added
        train_stats = train!(sol, online_oracle, batch; opt_state)
        ema_update!(ema_oracle, online_oracle, sol.ema_decay)
        if sol.lr_decay < 1f0
            Flux.Optimisers.adjust!(opt_state; eta = sol.lr * sol.lr_decay ^ update)
        end
        cb_oracle = ema_oracle
        cb_info = merge(
            (oracle=cb_oracle, iter=update, update, steps_done, max_steps=sol.max_steps, samples_added, online_oracle, ema_oracle),
            selfplay_metrics(hists),
            training_metrics(train_stats),
            (; minibatch_metrics=training_minibatch_metrics(train_stats)),
            batch_metrics(batch),
            oracle_metrics(ema_oracle, prev_ema_oracle, batch),
        )
        call(cb, cb_info)
        prev_ema_oracle = deepcopy(ema_oracle)
        # decay!(sol.optimiser, 0.9)
    end
    finish!(progress)
    planner_oracle = ema_oracle
    planner = AlphaZeroPlanner(
        game, planner_oracle;
        max_iter      = sol.mcts_params.tree_queries,
        max_time      = sol.mcts_params.max_time,
        max_depth     = sol.mcts_params.max_depth,
        search_style  = sol.mcts_params.search_style
    )
    return planner
end

function distributed_mcts(progress, game, mcts_params, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0)
    hists = NamedTuple[]
    collected = 0
    n_tasks = max(1, Distributed.nworkers())
    while collected < num_steps
        task_count = min(n_tasks, num_steps - collected)
        new_hists = pmap(1:task_count) do _
            mcts_sim(mcts_params, game, rand(s0); ϵ)
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

function serial_mcts(progress, game, mcts_params, num_steps::Int, s0; ϵ=0.30, steps_done::Int=0)
    hists = NamedTuple[]
    collected = 0
    while collected < num_steps
        hist = trim_history(mcts_sim(mcts_params, game, rand(s0); ϵ), num_steps - collected)
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
        policy = (hist.policy[1][1:n], hist.policy[2][1:n]),
    )
end

function merge_histories(hists)
    s = Vector{Float32}[]
    r = Float64[]
    v = Float64[]
    p1 = Vector{Float64}[]
    p2 = Vector{Float64}[]
    for hist in hists
        append!(s, hist.s)
        append!(r, hist.r)
        append!(v, hist.v)
        append!(p1, hist.policy[1])
        append!(p2, hist.policy[2])
    end
    return (; s, r, v, policy=(p1, p2))
end
