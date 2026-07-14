using Distributed
using Flux
using JLD2
using ProgressMeter
using Random
using MarkovGames

@testset "solver.jl and train.jl" begin
    ema_model = Fixtures.simple_fitted_regret_model()
    model = Fixtures.simple_fitted_regret_model()
    for p in Flux.trainables(ema_model)
        fill!(p, 0f0)
    end
    for p in Flux.trainables(model)
        fill!(p, 2f0)
    end
    AZ.ema_update!(ema_model, model, 0.25)
    @test all(all(isapprox.(p, 1.5f0; atol=1e-6)) for p in Flux.trainables(ema_model))

    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.TableOracle(values=Dict(1f0 => 0f0))
    search = AZ.SMOOSSearch(;
        oracle,
        oos_iterations=3,
        τ=2.0,
        transfer_weight=0.5,
        max_depth=5,
    )
    planner = AZ.AlphaZeroPlanner(game, search)
    @test planner.search === search
    @test !hasproperty(planner, :search_params)
    @test !hasproperty(planner, :smoos_params)
    @test !hasproperty(planner, :mcts_params)
    @test planner.search.oos_iterations == 3
    @test planner.search.τ == 2.0
    @test planner.search.transfer_weight == 0.5
    @test AZ.with_oracle(search, :new_oracle).oracle == :new_oracle
    @test AZ.with_oracle(search, :new_oracle; τ=4.0).τ == 4.0
    @test AZ.advance_transfer_tau(0.0, 3, 0.5) == 1.5
    @test AZ.advance_transfer_tau(1.5, 3, 0.5) == 2.25

    transfer_config = AZ.LossScaledTransfer(
        confidence_ema_decay=0.5,
        loss_tail_fraction=0.5,
    )
    confidence_stats = (;
        regret_losses=Float32[9, 9, 1, 1],
        strategy_losses=Float32[9, 9, 0.6, 0.6],
        zero_regret_losses=Float32[9, 9, 2, 2],
        strategy_entropy_losses=Float32[9, 9, 0.2, 0.2],
        uniform_strategy_losses=Float32[9, 9, 1.0, 1.0],
    )
    q_regret, q_strategy = AZ.transfer_fit_confidence(confidence_stats, transfer_config)
    @test q_regret ≈ 0.5
    @test isapprox(q_strategy, 0.5; atol=1e-6)
    adaptive_search = AZ.SMOOSSearch(
        oracle=oracle,
        oos_iterations=4,
        loss_scaled_transfer=transfer_config,
    )
    adaptive_state = AZ.initial_search_state(adaptive_search)
    adaptive_state = AZ.advance_search_state(adaptive_search, adaptive_state, confidence_stats)
    @test adaptive_state.source_mass == 4.0
    @test adaptive_state.regret_confidence ≈ 0.25
    @test isapprox(adaptive_state.strategy_confidence, 0.25; atol=1e-6)
    adaptive_callback = AZ.search_callback_state(adaptive_search, adaptive_state)
    @test adaptive_callback.transfer_source_mass == 4.0
    @test adaptive_callback.transfer_regret_confidence ≈ 0.25

    train_oracle = Fixtures.simple_fitted_regret_model()
    train_search = AZ.SMOOSSearch(; oos_iterations=0, max_depth=3, oracle=train_oracle)
    solver = AZ.AlphaZeroSolver(
        max_steps=2,
        num_steps=2,
        sim_depth=7,
        search=train_search,
        update_epochs=1,
        num_batches=1,
        ema_decay=0.5f0,
    )
    @test solver.search === train_search
    @test !hasproperty(solver, :search_params)
    @test !hasproperty(solver, :smoos_params)
    @test !hasproperty(solver, :mcts_params)
    @test !hasproperty(solver, :transfer_weight)
    @test !hasproperty(solver, :value_weight)
    @test !hasproperty(solver, :regret_weight)
    @test !hasproperty(solver, :strategy_weight)
    @test AZ.AlphaZeroPlanner(solver, game).search.oos_iterations == 0
    @test AZ.AlphaZeroPlanner(game, solver).search.max_depth == 3
    @test solver.sim_depth == 7
    @test solver.ema

    bounded_lr_solver = AZ.AlphaZeroSolver(
        search=train_search,
        lr=1f-3,
        lr_decay=0.5f0,
        lr_min=2f-4,
        lr_max=8f-4,
    )
    @test AZ.learning_rate(bounded_lr_solver, 0) == 8f-4
    @test AZ.learning_rate(bounded_lr_solver, 1) == 5f-4
    @test AZ.learning_rate(bounded_lr_solver, 10) == 2f-4

    dist, info = MarkovGames.behavior_info(planner, false)
    @test rand(Random.MersenneTwister(1), dist) isa Tuple
    @test hasproperty(info, :tree)
    @test hasproperty(info, :regret)
    @test hasproperty(info, :strategy)
    @test hasproperty(info, :v)

    mktempdir() do dir
        source = Fixtures.simple_fitted_regret_model()
        target = Fixtures.simple_fitted_regret_model()
        for p in Flux.trainables(target)
            fill!(p, -1f0)
        end

        state_path = joinpath(dir, "state.jld2")
        jldsave(state_path; model_state=Flux.state(source))
        Flux.loadmodel!(target, state_path)
        @test Flux.state(target) == Flux.state(source)

        planner_target = AZ.AlphaZeroPlanner(
            game,
            AZ.SMOOSSearch(; oracle=Fixtures.simple_fitted_regret_model(), oos_iterations=0, max_depth=1),
        )
        Flux.loadmodel!(planner_target, state_path)
        @test Flux.state(AZ.oracle(planner_target)) == Flux.state(source)

        oracle_path = joinpath(dir, "oracle.jld2")
        jldsave(oracle_path; oracle=source)
        @test AZ.load_oracle(dir) isa AZ.FittedRegretModel
        @test AZ.load_oracle(oracle_path) isa AZ.FittedRegretModel
    end

    batch = (
        s = [Float32[0], Float32[1], Float32[0], Float32[1]],
        v = Float32[1, 0, 1, 0],
        regret = (
            [Float32[0.5, -0.5], Float32[-0.25, 0.25], Float32[0.5, -0.5], Float32[-0.25, 0.25]],
            [Float32[-0.5, 0.5], Float32[0.25, -0.25], Float32[-0.5, 0.5], Float32[0.25, -0.25]],
        ),
        strategy = (
            [Float32[1, 0], Float32[0, 1], Float32[1, 0], Float32[0, 1]],
            [Float32[0, 1], Float32[1, 0], Float32[0, 1], Float32[1, 0]],
        ),
    )
    X, v_target, r_target, s_target = AZ.training_arrays(batch)
    @test size(X) == (1, 4)
    @test length(v_target) == 4
    @test size(r_target[1]) == (2, 4)
    @test size(s_target[1]) == (2, 4)
    @test sort(vcat(collect.(AZ.minibatches(collect(1:5), 2))...)) == collect(1:5)
    @test AZ.l2_penalty(Float32[1, 2, 3]) ≈ Float32(14 / 3)
    # oracle_metrics subsamples the batch with replacement via the global RNG;
    # seed so this 4-element batch draws a variance-nonzero value subset
    # regardless of upstream RNG state (explained variance is NaN on zero var).
    Random.seed!(1)
    oracle_stats = AZ.oracle_metrics(
        Fixtures.simple_fitted_regret_model(),
        Fixtures.simple_fitted_regret_model(),
        batch,
    )
    @test hasproperty(oracle_stats, :target_regret_l2)
    @test hasproperty(oracle_stats, :policy_kl_p1)
    @test hasproperty(oracle_stats, :value_explained_variance)
    @test !hasproperty(oracle_stats, :target_strategy_kl_p1)
    @test isfinite(oracle_stats.target_regret_l2)
    @test oracle_stats.target_regret_l2 > 0
    @test isfinite(oracle_stats.value_explained_variance)

    oracle2 = Fixtures.simple_fitted_regret_model()
    before = deepcopy(Flux.state(oracle2))
    train_sol = (
        num_batches = 2,
        update_epochs = 1,
        optimiser = Flux.Optimisers.Adam(0.01f0),
        rng = Random.MersenneTwister(1),
    )
    train_stats = AZ.train!(train_sol, oracle2, batch)
    @test length(train_stats.losses) == 2
    @test length(train_stats.value_losses) == 2
    @test length(train_stats.regret_losses) == 2
    @test length(train_stats.strategy_losses) == 2
    @test length(train_stats.zero_regret_losses) == 2
    @test length(train_stats.strategy_entropy_losses) == 2
    @test length(train_stats.uniform_strategy_losses) == 2
    @test hasproperty(AZ.training_metrics(train_stats), :mean_regret_loss)
    @test hasproperty(AZ.training_minibatch_metrics(train_stats), :strategy_loss)
    @test Flux.state(oracle2) != before

    value_only_oracle = Fixtures.simple_fitted_regret_model(;
        value_weight = 1.0f0,
        regret_weight = 0.0f0,
        strategy_weight = 0.0f0,
    )
    value_only_stats = AZ.train!(
        (
            num_batches = 1,
            update_epochs = 1,
            optimiser = Flux.Optimisers.Adam(0.0f0),
            rng = Random.MersenneTwister(2),
        ),
        value_only_oracle,
        batch,
    )
    @test only(value_only_stats.losses) ≈ only(value_only_stats.value_losses)

    policy_batch = (
        s = batch.s,
        v = batch.v,
        policy = batch.strategy,
    )
    @test_throws ArgumentError AZ.train!(train_sol, Fixtures.simple_fitted_regret_model(), policy_batch)
    @test_throws ArgumentError AZ.train!(train_sol, Fixtures.simple_actor_critic(), batch)

    @test AZ.lambda_gae_targets([1.0], [0.5], 0.0, 0.9, 0.0) ≈ [1.0]
    @test AZ.lambda_gae_targets([1.0], [0.5], 2.0, 0.9, 0.0) ≈ [2.8]
    @test AZ.lambda_gae_targets([1.0, 2.0], [0.0, 0.0], 3.0, 0.5, 1.0) ≈ [2.75, 3.5]

    progress = Progress(1; enabled=false)
    smoos_search = AZ.SMOOSSearch(oos_iterations=0, max_depth=2, oracle=Fixtures.simple_fitted_regret_model())
    serial = AZ.serial_smoos(progress, game, smoos_search, 1, initialstate(game); ϵ=0.0)
    @test length(serial) == 1

    distributed = AZ.distributed_smoos(Progress(1; enabled=false), game, smoos_search, 1, initialstate(game); ϵ=0.0)
    @test length(distributed) == 1

    callback_iters = Int[]
    callback_steps = Int[]
    callback_sim_depths = Int[]
    callback_transfer_taus = Float64[]
    callback_learning_rates = Float32[]
    callback_has_minibatches = Bool[]
    planner_out = MarkovGames.solve(solver, game; cb=info -> begin
        push!(callback_iters, info.iter)
        hasproperty(info, :steps_done) && push!(callback_steps, info.steps_done)
        hasproperty(info, :sim_depth) && push!(callback_sim_depths, info.sim_depth)
        hasproperty(info, :transfer_tau) && push!(callback_transfer_taus, info.transfer_tau)
        hasproperty(info, :learning_rate) && push!(callback_learning_rates, info.learning_rate)
        push!(callback_has_minibatches, hasproperty(info, :minibatch_metrics))
    end)
    @test planner_out isa AZ.AlphaZeroPlanner
    @test callback_steps[end] == 2
    @test callback_has_minibatches == [false, true]
    @test callback_iters == [0, 1]
    @test callback_sim_depths == [7, 7]
    @test callback_transfer_taus == [0.0, 0.0]
    @test callback_learning_rates == [solver.lr, solver.lr]
    @test planner_out.search isa AZ.SMOOSSearch

    no_ema_oracle = Fixtures.simple_fitted_regret_model()
    no_ema_solver = AZ.AlphaZeroSolver(
        max_steps=1,
        num_steps=1,
        sim_depth=1,
        search=AZ.SMOOSSearch(oos_iterations=0, max_depth=1, oracle=no_ema_oracle),
        update_epochs=1,
        num_batches=1,
        ema=false,
    )
    @test !no_ema_solver.ema
    no_ema_uses_online = Bool[]
    no_ema_reports_no_shadow = Bool[]
    MarkovGames.solve(no_ema_solver, game; cb=info -> begin
        push!(no_ema_uses_online, info.oracle === info.online_oracle)
        push!(no_ema_reports_no_shadow, isnothing(info.ema_oracle))
    end)
    @test all(no_ema_uses_online)
    @test all(no_ema_reports_no_shadow)
end
