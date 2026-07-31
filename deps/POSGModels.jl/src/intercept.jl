module Intercept

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra
using RecipesBase

export Coord, InterceptMG, InterceptState

const ACTION_DIRS = SA[
    SA[0, 1], # up
    SA[1, 0], # right
    SA[0,-1], # down
    SA[-1,0]  # left
]

const Coord = SVector{2, Int}

struct InterceptState
    attacker::Coord
    defender::Coord
    terminal::Bool
    InterceptState(attacker, defender, terminal=false) = new(attacker, defender, terminal)
end

Base.@kwdef struct InterceptMG{RM} <: MG{InterceptState, Tuple{Int,Int}}
    reward_model    ::  RM              = DenseReward()
    tag_reward      ::  Float64         = 10.0
    step_cost       ::  Float64         = 1.0
    discount        ::  Float64         = 0.95
    floor           ::  Coord           = Coord(11, 7)
    obstacles       ::  Set{Coord}      = Set{Coord}()
    initialstate    ::  InterceptState  = InterceptState(Coord(1,1), Coord(1,5), false)
    goal            ::  Set{Coord}      = Set{Coord}([Coord(11, 7)])
end

Base.@kwdef struct DenseReward
    scale   ::  Float64 = 1.0
    peak    ::  Float64 = 10.0
    d_max   ::  Float64 = 5.0
end

function (dr::DenseReward)(s1::Coord, s2::Coord)
    (;scale, peak, d_max) = dr
    d = norm(s2 .- s1, 2)
    return if iszero(d)
        peak
    elseif d > d_max
        0.0
    else
        (scale/d)
    end
end

struct SparseReward end

(sr::SparseReward)(s1::Coord, s2::Coord) = all(s1 .== s2)

function MarkovGames.states(p::InterceptMG)
    v = map(CartesianIndices((p.floor[1], p.floor[2], p.floor[1], p.floor[2]))) do c_i
        t = Tuple(c_i)
        return InterceptState(
            Coord(t[1], t[2]), 
            Coord(t[3], t[4]), 
            false
        )
    end
    return push!(vec(v), InterceptState(Coord(1,1), Coord(1,1), true)) 
end

MarkovGames.actions(p::InterceptMG) = (1:4, 1:4)

MarkovGames.discount(p::InterceptMG) = p.discount

MarkovGames.initialstate(p::InterceptMG) = Deterministic(p.initialstate)

function add_if_clear(floor, obstacles, s::Coord, a::Coord)
    sp = s .+ a
    if (any(sp .< Coord(1,1)) || any(sp .> floor) || sp ∈ obstacles)
        return s
    else
        return sp
    end
end

add_if_clear(floor::Coord, obstacles, s::Coord, a::Int) = add_if_clear(floor, obstacles, s, ACTION_DIRS[a])

function MarkovGames.transition(p::InterceptMG{DenseReward}, s, (a1, a2))
    (;floor, obstacles) = p
    next_attacker = add_if_clear(floor, obstacles, s.attacker, a1)
    next_defender = add_if_clear(floor, obstacles, s.defender, a2)
    return if s.attacker ∈ p.goal 
        Deterministic(InterceptState(Coord(1,1), Coord(1,1), true))
    else
        Deterministic(InterceptState(next_attacker, next_defender, false))
    end
end

function MarkovGames.transition(p::InterceptMG, s, (a1, a2))
    (;floor, obstacles) = p
    return if s.attacker == s.defender || s.attacker ∈ p.goal 
        Deterministic(InterceptState(Coord(1,1), Coord(1,1), true))
    else
        next_attacker = add_if_clear(floor, obstacles, s.attacker, a1)
        next_defender = add_if_clear(floor, obstacles, s.defender, a2)
        Deterministic(InterceptState(next_attacker, next_defender, false))
    end
end

MarkovGames.isterminal(::InterceptMG, s) = s.terminal

function MarkovGames.reward(p::InterceptMG, s::InterceptState, a)
    if isterminal(p, s)
        return SA[0.0, 0.0]
    else
        r = -p.reward_model(s.attacker, s.defender)
        if s.attacker ∈ p.goal
            r += 10.0
        end
        return SA[Float64(r), -Float64(r)]
    end
end

function stateindex(f::Coord, s::InterceptState)
    (;attacker, defender, terminal) = s
    nc, nr = f
    n_pos = (nr^2)*(nc^2)
    if terminal
        return n_pos+1
    else
        return LinearIndices((nc,nr,nc,nr))[attacker[1], attacker[2], defender[1], defender[2]]
    end
end

MarkovGames.stateindex(p::InterceptMG, s::InterceptState) = stateindex(p.floor, s)

function MarkovGames.convert_s(::Type{Vector{T}} , s::InterceptState, p::InterceptMG) where T
    (; floor) = p
    attacker = (s.attacker .- floor ./ 2) ./ floor
    defender = (s.defender .- floor ./ 2) ./ floor
    return T[
        attacker[1], attacker[2],
        defender[1], defender[2]
    ]
end

function MarkovGames.convert_s(::Type{InterceptState}, x::AbstractVector, p::InterceptMG)
    (; floor) = p
    attacker = round.(Int, (x[1:2] .* floor) .+ floor ./ 2)
    defender = round.(Int, (x[3:4] .* floor) .+ floor ./ 2)
    return InterceptState(Coord(attacker[1], attacker[2]), Coord(defender[1], defender[2]), false)
end

## visualization

function action_lines(x::Coord)
    return map(ACTION_DIRS) do a
        sp = x + a
        [x[1], sp[1]], [x[2], sp[2]]
    end
end

@recipe function f(game::InterceptMG, s::InterceptState)
    (;attacker, defender) = s
    xlims --> (0, game.floor[1]+1)
    ylims --> (0, game.floor[2]+1)
    xticks --> nothing
    yticks --> nothing
    goals = collect(game.goal)
    @series begin
        seriestype  := :scatter
        c           --> [:blue,:red]
        [attacker[1], defender[1]], [attacker[2], defender[2]]
    end
    @series begin
        seriestype := :scatter
        ms := 20
        c := :yellow
        first.(goals), last.(goals)
    end
end

@recipe function f(game::InterceptMG, s::InterceptState, σ1::AbstractVector, σ2::AbstractVector)
    (;attacker, defender) = s
    pol1 = σ1 |> permutedims
    pol2 = σ2 |> permutedims
    @series begin
        c       --> 1
        lw      --> 10
        alpha   --> pol1
        action_lines(attacker)
    end
    @series begin
        c       --> :red
        lw      --> 10
        alpha   --> pol2
        action_lines(defender)
    end
    @series begin
        game, s
    end
end

@recipe f(game::InterceptMG, s::InterceptState, σ1::SparseCat, σ2::SparseCat) = game, s, σ1.probs, σ2.probs

@recipe f(game::InterceptMG, s::InterceptState, σ::ProductDistribution) = game, s, σ[1], σ[2]

end
