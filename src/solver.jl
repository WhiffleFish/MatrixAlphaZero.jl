@proto @kwdef struct AlphaZeroSolver{OPT, RNG<:Random.AbstractRNG, O, MP}
    max_iter        ::  Int     = 100
    n_iter          ::  Int     = 200
    steps_per_iter  ::  Int     = 50_000
    buff_cap        ::  Int     = 1_000_000

    # MCTS
    oracle          ::  O
    mcts_params     ::  MP      = MCTSParams(;oracle)

    # Training args
    batchsize       ::  Int     = 128
    lr              ::  Float32 = 3f-4
    train_intensity ::  Int     = 6

    optimiser       ::  OPT     = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(1f0),
        Flux.Optimisers.ClipGrad(1f0),
        Flux.Optimisers.Adam(lr)
    )
    rng             ::  RNG = Random.default_rng()
end


@kwdef mutable struct AlphaZeroPlanner{G<:MG, Oracle} <: Policy
    game        ::  G
    oracle      ::  Oracle
    max_iter    ::  Int
    max_time    ::  Float64
    max_depth   ::  Int
    c           ::  Float64
end

function AlphaZeroPlanner(
        oracle::ActorCritic, 
        game::MG;
        max_iter    =   0,
        max_time    =   Inf,
        max_depth   =   typemax(Int),
        c           =   1.0
    )
    return AlphaZeroPlanner(;
        game, 
        oracle = oracle, 
        max_iter, 
        max_time, 
        max_depth, 
        c
    )
end

function Flux.loadmodel!(planner::AlphaZeroPlanner, path::String)
    Flux.loadmodel!(planner.oracle, JLD2.load(path)["model_state"])
end

function load_oracle(path)
    jldopen(joinpath(path, "oracle.jld2"))["oracle"]
    if isdir(path)
        return jldopen(joinpath(path, "oracle.jld2"))["oracle"]
    elseif isfile(path)
        return jldopen(path)["oracle"]
    else # when would this ever be hit? lmao
        error("$(path) is neither file nor dir")
    end
end

AlphaZeroPlanner(sol::AlphaZeroSolver, game::MG; kwargs...) = AlphaZeroPlanner(
    sol.oracle, game; 
    max_iter    =   sol.mcts_params.tree_queries, 
    max_time    =   sol.mcts_params.max_time,
    max_depth   =   sol.mcts_params.max_depth,
    c           =   sol.mcts_params.c,
    kwargs...
)

AlphaZeroPlanner(planner::AlphaZeroPlanner; kwargs...) = AlphaZeroPlanner(
    planner.oracle, 
    planner.game,
    max_iter    =   planner.tree_queries, 
    max_time    =   planner.max_time,
    max_depth   =   planner.max_depth,
    c           =   planner.c,
    kwargs...
)

function MarkovGames.behavior(policy::AlphaZeroPlanner, s)
    (;oracle, game, max_iter, max_time, max_depth, c) = policy
    A1, A2 = actions(game)
    x,y,v = if iszero(policy.max_depth) || iszero(policy.max_iter) || iszero(policy.max_time)
        solve(oracle_matrix_game(game, oracle, s))
    else
        search(MCTSParams(;oracle, tree_queries=max_iter, max_depth, max_time, c), game, s)
    end
    return ProductDistribution(SparseCat(A1,x), SparseCat(A2,y))
end

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG; s0=initialstate(game), cb=(info)->())
    distributed = Distributed.nprocs() > 1
    mcts_iter = sol.steps_per_iter ÷ sol.mcts_params.max_depth
    total_iter = mcts_iter * sol.max_iter
    progress = Progress(total_iter, safe_lock=false)
    buf = Buffer(sol.buff_cap)
    train_losses = Vector{Float32}[]
    value_losses = Vector{Float32}[]
    policy_losses = Vector{Float32}[]
    call(cb, (;oracle=sol.mcts_params.oracle, iter=0))
    for i ∈ 1:sol.max_iter
        temperature = sol.mcts_params.temperature(i)
        hists = if distributed
            distributed_mcts(progress, game, sol.mcts_params, mcts_iter, s0; temperature)
        else
            serial_mcts(progress, game, sol.mcts_params, mcts_iter, s0; temperature)
        end
        foreach(hists) do hist
            push!(buf, hist)
        end
        train_info = train!(sol, sol.mcts_params.oracle, buf)
        push!(train_losses, train_info[:losses])
        push!(value_losses, train_info[:value_losses])
        push!(policy_losses, train_info[:policy_losses])
        call(cb, (;oracle=sol.mcts_params.oracle, iter=i))
    end
    finish!(progress)
    return AlphaZeroPlanner(sol, game), (;
        train_losses, value_losses, policy_losses, buffer=buf
    )
end

function distributed_mcts(progress, game, mcts_params, mcts_iter, s0; temperature=1.0)
    # https://discourse.julialang.org/t/does-anyone-have-a-progress-bar-for-pmap/11729/5
    channel = RemoteChannel(()->Channel{Bool}(), 1)
    @async while take!(channel)
        next!(progress)
    end
    hists = pmap(1:mcts_iter) do i
        hist = mcts_sim(mcts_params, game, rand(s0); temperature)
        put!(channel, true)
        return hist
    end
    put!(channel, false)
    return hists
end

function serial_mcts(progress, game, mcts_params, mcts_iter, s0; temperature=1.0)
    return map(1:mcts_iter) do i
        hist = mcts_sim(mcts_params, game, rand(s0); temperature)
        next!(progress)
        return hist
    end
end

function oracle_matrix_game(game, oracle, s)
    γ = discount(game)
    A1, A2 = actions(game)
    mat = zeros(length(A1), length(A2))
    for (i, a1) ∈ enumerate(A1)
        for (j, a2) ∈ enumerate(A2)
            a = (a1, a2)
            sp, r = @gen(:sp, :r)(game, s, a)
            vp = only(value(oracle, MarkovGames.convert_s(Vector{Float32}, sp, game)))
            mat[i,j] = r + γ * vp
        end
    end
    return mat
end
