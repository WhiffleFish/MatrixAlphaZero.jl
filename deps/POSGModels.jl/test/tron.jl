using POSGModels
using MarkovGames
using Random
using StaticArrays
using Test

using POSGModels.Tron: JointTronState, TronMG

function deterministic_value(d)
    return rand(Random.MersenneTwister(1), d)
end

@testset "TronMG" begin
    @testset "constructor initializes expected default state" begin
        game = TronMG(width=7, height=5)
        s0 = deterministic_value(initialstate(game))

        @test s0 isa JointTronState
        @test game.width == 7
        @test game.height == 5
        @test collect(actions(game)[1]) == [1, 2, 3]
        @test collect(actions(game)[2]) == [1, 2, 3]
        @test s0.p1 == [3, 3]
        @test s0.p2 == [5, 3]
        @test s0.h1 == 2
        @test s0.h2 == 4
        @test !isterminal(game, s0)
        @test count(==(1.0), MarkovGames.convert_s(Vector{Float64}, s0, game)[9:end]) == 2
    end

    @testset "straight move updates both players and trails" begin
        game = TronMG(width=9, height=5)
        s0 = deterministic_value(initialstate(game))
        sp = deterministic_value(transition(game, s0, (2, 2)))
        x = MarkovGames.convert_s(Vector{Float64}, sp, game)

        @test sp.p1 == [4, 3]
        @test sp.p2 == [6, 3]
        @test sp.h1 == 2
        @test sp.h2 == 4
        @test !isterminal(game, sp)
        @test reward(game, s0, (2, 2), sp) == SA[0.0, 0.0]
        @test count(==(1.0), x[9:end]) == 4
    end

    @testset "head-on collision is a draw" begin
        game = TronMG(width=5, height=3, p1_start=(2, 2), p2_start=(4, 2), headings=(2, 4))
        s0 = deterministic_value(initialstate(game))
        sp = deterministic_value(transition(game, s0, (2, 2)))

        @test isterminal(game, sp)
        @test sp.outcome == 0
        @test sp.p1 == [3, 2]
        @test sp.p2 == [3, 2]
        @test reward(game, s0, (2, 2), sp) == SA[0.0, 0.0]
    end

    @testset "wall collision gives surviving player the win" begin
        game = TronMG(width=3, height=3, p1_start=(3, 2), p2_start=(2, 3), headings=(2, 4), win_reward=2.5)
        s0 = deterministic_value(initialstate(game))
        sp = deterministic_value(transition(game, s0, (2, 2)))

        @test isterminal(game, sp)
        @test sp.outcome == -1
        @test sp.p1 == [3, 2]
        @test sp.p2 == [1, 3]
        @test reward(game, s0, (2, 2), sp) == SA[-2.5, 2.5]
    end

    @testset "state encoding shape and headings are stable" begin
        game = TronMG(width=5, height=4, p1_start=(2, 2), p2_start=(4, 3), headings=(1, 3))
        s0 = deterministic_value(initialstate(game))
        x = MarkovGames.convert_s(Vector{Float64}, s0, game)

        @test length(x) == 8 + 2 * game.width * game.height
        @test x[3:4] == [0.0, 1.0]
        @test x[7:8] == [0.0, -1.0]
        @test sum(x[9:end]) == 2.0
    end
end
