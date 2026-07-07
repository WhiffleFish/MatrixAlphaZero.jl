using Flux
using MarkovGames
using POMDPs
using ProgressMeter
using Random

@testset "CriticOnly value-only oracle" begin
    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.simple_critic_only()

    # No actor: state_policy is a uniform prior, only used as a zero-visit fallback.
    A1, A2 = actions(game)
    x, y = AZ.state_policy(oracle, game, false)
    @test x ≈ fill(inv(length(A1)), length(A1))
    @test y ≈ fill(inv(length(A2)), length(A2))
    @test sum(x) ≈ 1
    @test sum(y) ≈ 1

    v = AZ.state_value(oracle, game, false)
    @test length(v) == 1
    bv = AZ.batch_state_value(oracle, game, [false, false, false])
    @test size(bv, 2) == 3

    search = AZ.MCTSSearch(
        tree_queries=4,
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

    # Self-play still uses the policy-target sim path; those targets are simply
    # ignored by the value-only train!.
    progress = Progress(1; enabled=false)
    hists = AZ.serial_mcts(progress, game, search, 1, initialstate(game); ϵ=0.0, sim_depth=1)
    batch = AZ.merge_histories(hists)
    @test hasproperty(batch, :policy)
    @test !hasproperty(batch, :regret)

    stats = AZ.train!(solver, oracle, batch)
    @test length(stats.losses) == 1
    @test length(stats.value_losses) == 1
    @test !hasproperty(stats, :policy_losses)
    @test !hasproperty(stats, :regret_losses)

    train_metrics = AZ.training_metrics(stats)
    @test hasproperty(train_metrics, :mean_value_loss)
    @test !hasproperty(train_metrics, :mean_policy_loss)
    @test !hasproperty(train_metrics, :mean_regret_loss)

    minibatch_metrics = AZ.training_minibatch_metrics(stats)
    @test hasproperty(minibatch_metrics, :value_loss)
    @test !hasproperty(minibatch_metrics, :policy_loss)
    @test !hasproperty(minibatch_metrics, :regret_loss)

    oracle_stats = AZ.oracle_metrics(oracle, deepcopy(oracle), batch)
    @test isfinite(oracle_stats.value_pred_mse)
    @test hasproperty(oracle_stats, :value_explained_variance)
    @test !hasproperty(oracle_stats, :target_policy_kl_p1)
    @test !hasproperty(oracle_stats, :target_regret_l2)

    # End-to-end solve exercises the full self-play → train! → callback loop.
    planner_out = MarkovGames.solve(solver, game)
    @test planner_out.search isa AZ.MCTSSearch
    @test AZ.oracle(planner_out.search) isa AZ.CriticOnly
end
