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
    planner = AZ.AlphaZeroPlanner(
        game,
        oracle;
        oos_iterations=3,
        τ=2.0,
        max_depth=5,
    )
    @test planner.smoos_params.oos_iterations == 3
    @test planner.smoos_params.τ == 2.0

    params = AZ.SMOOSParams(planner)
    @test params.oos_iterations == 3
    @test params.τ == 2.0
    @test params.max_depth == 5
    @test AZ.with_oracle(params, :new_oracle).oracle == :new_oracle
    @test AZ.with_oracle(params, :new_oracle; τ=4.0).τ == 4.0
    @test AZ.advance_transfer_tau(0.0, 3, 0.5) == 1.5
    @test AZ.advance_transfer_tau(1.5, 3, 0.5) == 2.25

    train_oracle = Fixtures.simple_fitted_regret_model()
    solver = AZ.AlphaZeroSolver(
        max_steps=2,
        num_steps=2,
        sim_depth=7,
        oracle=train_oracle,
        smoos_params=AZ.SMOOSParams(oos_iterations=0, max_depth=3, oracle=train_oracle),
        update_epochs=1,
        num_batches=1,
        ema_decay=0.5f0,
    )
    @test AZ.AlphaZeroPlanner(solver, game).smoos_params.oos_iterations == 0
    @test AZ.AlphaZeroPlanner(game, solver).smoos_params.max_depth == 3
    @test solver.sim_depth == 7

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

        planner_target = AZ.AlphaZeroPlanner(game, Fixtures.simple_fitted_regret_model(); oos_iterations=0, max_depth=1)
        Flux.loadmodel!(planner_target, state_path)
        @test Flux.state(planner_target.oracle) == Flux.state(source)

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
    oracle_stats = AZ.oracle_metrics(
        Fixtures.simple_fitted_regret_model(),
        Fixtures.simple_fitted_regret_model(),
        batch,
    )
    @test hasproperty(oracle_stats, :target_regret_l2)
    @test isfinite(oracle_stats.target_regret_l2)
    @test oracle_stats.target_regret_l2 > 0

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
    @test Flux.state(oracle2) != before

    @test AZ.lambda_gae_targets([1.0], [0.5], 0.0, 0.9, 0.0) ≈ [1.0]
    @test AZ.lambda_gae_targets([1.0], [0.5], 2.0, 0.9, 0.0) ≈ [2.8]
    @test AZ.lambda_gae_targets([1.0, 2.0], [0.0, 0.0], 3.0, 0.5, 1.0) ≈ [2.75, 3.5]

    progress = Progress(1; enabled=false)
    smoos_params = AZ.SMOOSParams(oos_iterations=0, max_depth=2, oracle=Fixtures.simple_fitted_regret_model())
    serial = AZ.serial_smoos(progress, game, smoos_params, 1, initialstate(game); ϵ=0.0)
    @test length(serial) == 1

    distributed = AZ.distributed_smoos(Progress(1; enabled=false), game, smoos_params, 1, initialstate(game); ϵ=0.0)
    @test length(distributed) == 1

    callback_iters = Int[]
    callback_steps = Int[]
    callback_sim_depths = Int[]
    callback_has_minibatches = Bool[]
    planner_out = MarkovGames.solve(solver, game; cb=info -> begin
        push!(callback_iters, info.iter)
        hasproperty(info, :steps_done) && push!(callback_steps, info.steps_done)
        hasproperty(info, :sim_depth) && push!(callback_sim_depths, info.sim_depth)
        push!(callback_has_minibatches, hasproperty(info, :minibatch_metrics))
    end)
    @test planner_out isa AZ.AlphaZeroPlanner
    @test callback_steps[end] == 2
    @test callback_has_minibatches == [false, true]
    @test callback_iters == [0, 1]
    @test callback_sim_depths == [7, 7]
end
