@kwdef struct AlphaZeroSolver{OPT, RNG<:Random.AbstractRNG}
    max_iter        ::  Int     = 100
    n_iter          ::  Int     = 200
    steps_per_iter  ::  Int     = 50_000
    buff_cap        ::  Int     = 1_000_000

    # MCTS
    tree_queries    ::  Int     = 10
    c               ::  Float64 = 1.0

    # Training args
    batchsize       ::  Int     = 128
    lr              ::  Float32 = 3f-4

    optimiser       ::  OPT     = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(1f0),
        Flux.Optimisers.ClipGrad(1f0),
        Flux.Optimisers.Adam(lr)
    )
    rng::RNG = Random.default_rng()
end


struct AlphaZeroPlanner

end

function MarkovGames.solve(sol::AlphaZeroSolver, game::MG)
    p = Progress(sol.max_iter)

    tree = Tree(game)

    for i ∈ 1:sol.max_iter

        next!(p)
    end
    return AlphaZeroPlanner()
end
