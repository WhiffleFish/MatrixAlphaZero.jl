using Flux
using MarkovGames
using POMDPs
using ProgressMeter
using Random

@testset "MCTS solver integration" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.simple_actor_critic()
    search = AZ.MCTSSearch(
        tree_queries=0,
        max_depth=2,
        oracle=oracle,
    )
    solver = AZ.AlphaZeroSolver(
        max_steps=2,
        num_steps=2,
        sim_depth=1,
        search=search,
        update_epochs=1,
        num_batches=1,
        optimiser=Flux.Optimisers.Adam(0.0f0),
        rng=Random.MersenneTwister(11),
    )

    @test solver.search === search
    @test solver.search isa AZ.MCTSSearch
    @test !hasproperty(solver, :search_params)
    @test !hasproperty(solver, :smoos_params)
    @test !hasproperty(solver, :mcts_params)
    @test !hasproperty(solver, :transfer_weight)
    @test !hasproperty(solver.search, :transfer_weight)
    @test AZ.AlphaZeroPlanner(solver, game).search isa AZ.MCTSSearch
    @test AZ.AlphaZeroPlanner(game, solver).search.max_depth == 2

    planner = AZ.AlphaZeroPlanner(game, search)
    @test planner.search === search
    @test !hasproperty(planner, :search_params)
    @test !hasproperty(planner, :smoos_params)
    @test !hasproperty(planner, :mcts_params)
    updated_planner = AZ.AlphaZeroPlanner(planner; search=AZ.MCTSSearch(; tree_queries=3, max_depth=2, oracle))
    @test updated_planner.search.tree_queries == 3

    dist, info = MarkovGames.behavior_info(planner, false)
    @test rand(Random.MersenneTwister(1), dist) isa Tuple
    @test hasproperty(info, :tree)
    @test hasproperty(info, :policy)
    @test hasproperty(info, :v)
    @test !hasproperty(info, :regret)

    progress = Progress(1; enabled=false)
    hists = AZ.serial_mcts(progress, game, search, 1, initialstate(game); ϵ=0.0, sim_depth=1)
    batch = AZ.merge_histories(hists)
    @test hasproperty(batch, :policy)
    @test !hasproperty(batch, :regret)
    X, v_target, p_target = AZ.policy_training_arrays(batch)
    @test size(X, 2) == length(v_target) == 1
    @test size(p_target[1], 2) == 1

    stats = AZ.train!(solver, oracle, batch)
    @test length(stats.losses) == 1
    @test length(stats.value_losses) == 1
    @test length(stats.policy_losses) == 1
    @test !hasproperty(stats, :regret_losses)
    train_metrics = AZ.training_metrics(stats)
    minibatch_metrics = AZ.training_minibatch_metrics(stats)
    @test hasproperty(train_metrics, :mean_policy_loss)
    @test !hasproperty(train_metrics, :mean_regret_loss)
    @test hasproperty(minibatch_metrics, :policy_loss)
    @test !hasproperty(minibatch_metrics, :regret_loss)
    oracle_stats = AZ.oracle_metrics(oracle, deepcopy(oracle), batch)
    @test !hasproperty(oracle_stats, :target_regret_l2)
    @test hasproperty(oracle_stats, :target_policy_kl_p1)
    @test isfinite(oracle_stats.value_pred_mse)
    @test hasproperty(oracle_stats, :value_explained_variance)

    callback_has_transfer_tau = Bool[]
    callback_has_minibatches = Bool[]
    planner_out = MarkovGames.solve(solver, game; cb=info -> begin
        push!(callback_has_transfer_tau, hasproperty(info, :transfer_tau))
        push!(callback_has_minibatches, hasproperty(info, :minibatch_metrics))
    end)
    @test planner_out.search isa AZ.MCTSSearch
    @test callback_has_transfer_tau == [false, false]
    @test callback_has_minibatches == [false, true]
end
