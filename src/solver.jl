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


@proto struct AlphaZeroPlanner{Oracle}
    oracle::Oracle
end

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG)
    mcts_iter = sol.steps_per_iter ÷ sol.mcts_params.max_depth
    total_iter = mcts_iter * sol.max_iter
    p = Progress(total_iter)
    buf = Buffer(sol.buff_cap)
    for i ∈ 1:sol.max_iter
        for _ ∈ 1:mcts_iter
            hist = mcts_sim(sol.mcts_params, game, rand(initialstate(game)))
            push!(buf, hist)
            next!(p)
        end
        train!(sol, sol.mcts_params.oracle, buf)
    end
    return AlphaZeroPlanner(sol.mcts_params.oracle)
end
