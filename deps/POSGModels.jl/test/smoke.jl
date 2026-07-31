using POSGModels
using MarkovGames
using Random
using StaticArrays
using Test

function smoke_deterministic_value(d)
    return rand(Random.MersenneTwister(1), d)
end

function first_joint_action(game)
    return map(first, actions(game))
end

function smoke_mg(game)
    s = smoke_deterministic_value(initialstate(game))
    a = first_joint_action(game)
    sp, r = @gen(:sp, :r)(game, s, a, Random.MersenneTwister(2))

    @test sp !== nothing
    @test r isa SVector{length(players(game)), Float64}
    @test all(isfinite, r)
    @test isterminal(game, sp) isa Bool
end

@testset "game smoke tests" begin
    @testset "ContinuousTag" begin
        game = POSGModels.ContinuousTag(dtype=Float64, transition_noise=0.0)
        smoke_mg(game)

        s = smoke_deterministic_value(initialstate(game))
        a = first_joint_action(game)
        sp, o, r = @gen(:sp, :o, :r)(game, s, a, Random.MersenneTwister(3))

        @test sp !== nothing
        @test o isa Tuple{Int, Int}
        @test r isa SVector{2, Float64}
        @test length.(observations(game)) == (4, 4)
    end

    @testset "DiscreteTag" begin
        smoke_mg(POSGModels.DiscreteTag.TagMG())
    end

    @testset "SimpleGame" begin
        smoke_mg(POSGModels.SimpleGame.SimpleMG(2, 2; max_depth=1))
    end

    @testset "Intercept" begin
        smoke_mg(POSGModels.Intercept.InterceptMG())
    end

    @testset "StackedIntercept" begin
        smoke_mg(POSGModels.StackedIntercept.StackedInterceptMG())
    end

    @testset "Dubin" begin
        smoke_mg(POSGModels.Dubin.DubinMG())
    end

    @testset "Tron" begin
        smoke_mg(POSGModels.Tron.TronMG(width=7, height=5))
    end
end
