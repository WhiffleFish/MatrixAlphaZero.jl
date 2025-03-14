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


@proto struct AlphaZeroPlanner{G<:MG, Oracle}
    game::G
    oracle::Oracle
end

function behavior(policy::AlphaZeroPlanner, s)
    (;oracle, game) = policy
    A1, A2 = actions(game)
    x,y,v = solve(oracle_matrix_game(game, oracle, s))
    return SparseCat(A1,x), SparseCat(A2,y)
end

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG; s0=initialstate(game), cb=(info)->())
    distributed = Distributed.nprocs() > 1
    mcts_iter = sol.steps_per_iter ÷ sol.mcts_params.max_depth
    total_iter = mcts_iter * sol.max_iter
    progress = Progress(total_iter, safe_lock=false)
    buf = Buffer(sol.buff_cap)
    train_losses = Vector{Float32}[]
    for i ∈ 1:sol.max_iter
        hists = if distributed
            distributed_mcts(progress, game, sol.mcts_params, mcts_iter, s0)
        else
            serial_mcts(progress, game, sol.mcts_params, mcts_iter, s0)
        end
        foreach(hists) do hist
            push!(buf, hist)
        end
        train_info = train!(sol, sol.mcts_params.oracle, buf)
        push!(train_losses, train_info[:losses])
        call(cb, (;oracle=sol.mcts_params.oracle, iter=i))
    end
    finish!(progress)
    return AlphaZeroPlanner(game, sol.mcts_params.oracle), (;
        train_losses, buffer=buf
    )
end

function distributed_mcts(progress, game, mcts_params, mcts_iter, s0)
    # https://discourse.julialang.org/t/does-anyone-have-a-progress-bar-for-pmap/11729/5
    channel = RemoteChannel(()->Channel{Bool}(), 1)
    @async while take!(channel)
        next!(progress)
    end
    hists = pmap(1:mcts_iter) do i
        hist = mcts_sim(mcts_params, game, rand(s0))
        put!(channel, true)
        return hist
    end
    put!(channel, false)
    return hists
end

function serial_mcts(progress, game, mcts_params, mcts_iter, s0)
    return map(1:mcts_iter) do i
        hist = mcts_sim(mcts_params, game, rand(s0))
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
            vp = only(oracle(MarkovGames.convert_s(Vector{Float32}, sp, game)))
            mat[i,j] = r + γ * vp
        end
    end
    return mat
end
