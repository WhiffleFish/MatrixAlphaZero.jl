using Distributed
using Flux
using JLD2
using ProgressMeter
using Random

@testset "solver.jl and train.jl" begin
    ema_model = Fixtures.simple_actor_critic()
    model = Fixtures.simple_actor_critic()
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
    planner = AZ.AlphaZeroPlanner(game, oracle; max_iter=3, max_time=4.0, max_depth=5, c=1.2, matrix_solver=Fixtures.GreedyMatrixSolver())
    @test planner.max_iter == 3

    params = AZ.MCTSParams(planner)
    @test params.tree_queries == 3
    @test params.max_depth == 5
    @test params.c == 1.2
    @test AZ.with_oracle(params, :new_oracle).oracle == :new_oracle

    train_oracle = Fixtures.simple_actor_critic()
    solver = AZ.AlphaZeroSolver(
        max_iter=1,
        steps_per_iter=2,
        buff_cap=16,
        oracle=train_oracle,
        mcts_params=AZ.MCTSParams(tree_queries=0, max_depth=2, matrix_solver=Fixtures.GreedyMatrixSolver(), oracle=train_oracle),
        batchsize=1,
        train_intensity=1,
        ema_decay=0.5f0,
    )
    @test AZ.AlphaZeroPlanner(solver, game).max_iter == 0
    @test AZ.AlphaZeroPlanner(game, solver).max_depth == 2
    @test AZ.AlphaZeroPlanner(planner; c=2.0).c == 2.0

    mat = AZ.oracle_matrix_game(game, oracle, false)
    @test mat == game.rewards

    dist, info = MarkovGames.behavior_info(planner, false)
    @test rand(Random.MersenneTwister(1), dist) isa Tuple
    @test hasproperty(info, :tree)
    @test hasproperty(info, :v)

    mktempdir() do dir
        source = Fixtures.simple_actor_critic()
        target = Fixtures.simple_actor_critic()
        for p in Flux.trainables(target)
            fill!(p, -1f0)
        end

        state_path = joinpath(dir, "state.jld2")
        jldsave(state_path; model_state=Flux.state(source))
        Flux.loadmodel!(target, state_path)
        @test Flux.state(target) == Flux.state(source)

        planner_target = AZ.AlphaZeroPlanner(game, Fixtures.simple_actor_critic(); max_iter=0, max_depth=1, max_time=1.0, c=1.0, matrix_solver=Fixtures.GreedyMatrixSolver())
        Flux.loadmodel!(planner_target, state_path)
        @test Flux.state(planner_target.oracle) == Flux.state(source)

        oracle_path = joinpath(dir, "oracle.jld2")
        jldsave(oracle_path; oracle=source)
        @test AZ.load_oracle(dir) isa AZ.ActorCritic
        @test AZ.load_oracle(oracle_path) isa AZ.ActorCritic
    end

    buf = AZ.Buffer(8)
    push!(buf, (
        s = [Float32[0], Float32[1], Float32[0], Float32[1]],
        v = Float32[1, 0, 1, 0],
        policy = (
            [Float32[1, 0], Float32[0, 1], Float32[1, 0], Float32[0, 1]],
            [Float32[0, 1], Float32[1, 0], Float32[0, 1], Float32[1, 0]],
        ),
    ))
    X, v_target, p_target = AZ.get_batch(buf, 2)
    @test size(X) == (1, 2)
    @test length(v_target) == 2
    @test size(p_target[1]) == (2, 2)
    @test AZ.l2_penalty(Float32[1, 2, 3]) ≈ Float32(14 / 3)

    oracle2 = Fixtures.simple_actor_critic()
    before = deepcopy(Flux.state(oracle2))
    train_sol = (
        batchsize = 2,
        steps_per_iter = 4,
        train_intensity = 1,
        optimiser = Flux.Optimisers.Adam(0.01f0),
    )
    train_info = AZ.train!(train_sol, oracle2, buf)
    @test length(train_info.losses) == 2
    @test length(train_info.value_losses) == 2
    @test length(train_info.policy_losses) == 2
    @test Flux.state(oracle2) != before

    progress = Progress(1; enabled=false)
    mcts_params = AZ.MCTSParams(tree_queries=0, max_depth=2, matrix_solver=Fixtures.GreedyMatrixSolver(), oracle=Fixtures.simple_actor_critic())
    serial = AZ.serial_mcts(progress, game, mcts_params, 1, initialstate(game); ϵ=0.0)
    @test length(serial) == 1

    distributed = AZ.distributed_mcts(Progress(1; enabled=false), game, mcts_params, 1, initialstate(game); ϵ=0.0)
    @test length(distributed) == 1

    callback_iters = Int[]
    planner_out, solve_info = MarkovGames.solve(solver, game; cb=info -> push!(callback_iters, info.iter))
    @test planner_out isa AZ.AlphaZeroPlanner
    @test length(solve_info.buffer) > 0
    @test callback_iters == [0, 1]
end
