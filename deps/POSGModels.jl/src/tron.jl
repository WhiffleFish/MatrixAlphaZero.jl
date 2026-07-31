module Tron

using MarkovGames
using StaticArrays
using POMDPTools
using RecipesBase

export Cell, JointTronState, TronMG

const Cell = SVector{2, Int}

# Heading convention:
#   1 => up
#   2 => right
#   3 => down
#   4 => left
const DIRS = (
    SA[0, 1],
    SA[1, 0],
    SA[0, -1],
    SA[-1, 0],
)

"""
State for simultaneous Tron / light-cycle.

Fields
------
- `p1`, `p2`: current head positions
- `h1`, `h2`: current headings in {1,2,3,4}
- `trail1`, `trail2`: bitboards for occupied cells by each player
- `terminal`: whether the game is over
- `outcome`: +1 if player 1 won, -1 if player 2 won, 0 otherwise / draw
"""
struct JointTronState
    p1::Cell
    p2::Cell
    h1::Int
    h2::Int
    trail1::UInt128
    trail2::UInt128
    terminal::Bool
    outcome::Int8
    function JointTronState(
        p1::Cell, p2::Cell,
        h1::Integer, h2::Integer,
        trail1::UInt128, trail2::UInt128,
        terminal::Bool=false,
        outcome::Integer=0,
    )
        new(p1, p2, Int(h1), Int(h2), trail1, trail2, terminal, Int8(outcome))
    end
end

"""
Deterministic simultaneous-action Tron benchmark.

By default, each player has three relative actions:
    1 => turn left
    2 => go straight
    3 => turn right

The board is encoded as a UInt128 bitboard, so width*height must be <= 128.
"""
struct TronMG <: MG{JointTronState, Tuple{Int, Int}}
    width::Int
    height::Int
    actions::NTuple{2, Vector{Int}}   # relative heading changes, e.g. [-1, 0, +1]
    win_reward::Float64
    draw_reward::Float64
    discount::Float64
    initialstate::JointTronState
end

# -----------------------------
# Constructors / utilities
# -----------------------------

tocell(x) = SA[Int(x[1]), Int(x[2])]

n_cells(game::TronMG) = game.width * game.height

inbounds(game::TronMG, p::Cell) = 1 <= p[1] <= game.width && 1 <= p[2] <= game.height

function linidx(game::TronMG, p::Cell)
    @assert inbounds(game, p)
    return p[1] + (p[2] - 1) * game.width
end

function cell_from_idx(game::TronMG, idx::Int)
    @assert 1 <= idx <= n_cells(game)
    x = 1 + (idx - 1) % game.width
    y = 1 + (idx - 1) ÷ game.width
    return SA[x, y]
end

cellbit(game::TronMG, p::Cell) = UInt128(1) << (linidx(game, p) - 1)

occupied(trails::UInt128, bit::UInt128) = !iszero(trails & bit)

turn_heading(h::Int, δ::Int) = mod1(h + δ, 4)

step_forward(p::Cell, h::Int) = p + DIRS[h]

function default_initial_state(
    width::Int,
    height::Int,
    p1::Cell,
    p2::Cell,
    h1::Int,
    h2::Int,
)
    width * height <= 128 || error("TronMG currently requires width*height <= 128.")
    1 <= h1 <= 4 || error("Player 1 heading must be in 1:4.")
    1 <= h2 <= 4 || error("Player 2 heading must be in 1:4.")
    1 <= p1[1] <= width || error("Player 1 start x-position out of bounds.")
    1 <= p1[2] <= height || error("Player 1 start y-position out of bounds.")
    1 <= p2[1] <= width || error("Player 2 start x-position out of bounds.")
    1 <= p2[2] <= height || error("Player 2 start y-position out of bounds.")
    p1 == p2 && error("Players must start in different cells.")

    dummy = TronMG(
        width,
        height,
        (Int[-1, 0, 1], Int[-1, 0, 1]),
        1.0,
        0.0,
        0.99,
        JointTronState(SA[1, 1], SA[2, 1], 1, 1, UInt128(0), UInt128(0), false, 0),
    )
    trail1 = cellbit(dummy, p1)
    trail2 = cellbit(dummy, p2)
    return JointTronState(p1, p2, h1, h2, trail1, trail2, false, 0)
end

function TronMG(;
    width::Int=10,
    height::Int=10,
    actions::NTuple{2, Vector{Int}}=(Int[-1, 0, 1], Int[-1, 0, 1]),
    win_reward::Real=1.0,
    draw_reward::Real=0.0,
    discount::Real=0.99,
    p1_start=nothing,
    p2_start=nothing,
    headings::NTuple{2, Int}=(2, 4),
)
    width * height <= 128 || error("TronMG currently requires width*height <= 128.")

    p1 = isnothing(p1_start) ? SA[3, cld(height, 2)] : tocell(p1_start)
    p2 = isnothing(p2_start) ? SA[width - 2, cld(height, 2)] : tocell(p2_start)

    # temporary game so we can build the initial state's bitboards consistently
    tmp = TronMG(
        width,
        height,
        actions,
        Float64(win_reward),
        Float64(draw_reward),
        Float64(discount),
        JointTronState(SA[1, 1], SA[2, 1], 1, 1, UInt128(0), UInt128(0), false, 0),
    )

    inbounds(tmp, p1) || error("Player 1 start is out of bounds.")
    inbounds(tmp, p2) || error("Player 2 start is out of bounds.")
    p1 == p2 && error("Players must start in different cells.")

    h1, h2 = headings
    1 <= h1 <= 4 || error("Player 1 heading must be in 1:4.")
    1 <= h2 <= 4 || error("Player 2 heading must be in 1:4.")

    s0 = JointTronState(
        p1,
        p2,
        h1,
        h2,
        cellbit(tmp, p1),
        cellbit(tmp, p2),
        false,
        0,
    )

    return TronMG(
        width,
        height,
        actions,
        Float64(win_reward),
        Float64(draw_reward),
        Float64(discount),
        s0,
    )
end

# -----------------------------
# MarkovGames interface
# -----------------------------

MarkovGames.actions(game::TronMG) = eachindex.(game.actions)

MarkovGames.discount(game::TronMG) = game.discount

MarkovGames.initialstate(game::TronMG) = Deterministic(game.initialstate)

MarkovGames.isterminal(::TronMG, s::JointTronState) = s.terminal

function MarkovGames.transition(game::TronMG, s::JointTronState, a::Tuple{Int, Int})
    if isterminal(game, s)
        return Deterministic(s)
    end

    a1, a2 = a
    δ1 = game.actions[1][a1]
    δ2 = game.actions[2][a2]

    h1p = turn_heading(s.h1, δ1)
    h2p = turn_heading(s.h2, δ2)

    p1p = step_forward(s.p1, h1p)
    p2p = step_forward(s.p2, h2p)

    occ = s.trail1 | s.trail2

    crash1 = !inbounds(game, p1p)
    crash2 = !inbounds(game, p2p)

    if !crash1
        crash1 = occupied(occ, cellbit(game, p1p))
    end
    if !crash2
        crash2 = occupied(occ, cellbit(game, p2p))
    end

    # simultaneous head-on into the same fresh cell
    if !crash1 && !crash2 && p1p == p2p
        crash1 = true
        crash2 = true
    end

    # for plotting / terminal display, keep out-of-bounds players at their last valid position
    p1_disp = inbounds(game, p1p) ? p1p : s.p1
    p2_disp = inbounds(game, p2p) ? p2p : s.p2

    trail1p = s.trail1
    trail2p = s.trail2

    # if a player survives, their next cell becomes part of the trail
    if !crash1
        trail1p |= cellbit(game, p1p)
    end
    if !crash2
        trail2p |= cellbit(game, p2p)
    end

    if crash1 || crash2
        outcome = crash1 == crash2 ? 0 : (crash1 ? -1 : 1)
        sp = JointTronState(p1_disp, p2_disp, h1p, h2p, trail1p, trail2p, true, outcome)
        return Deterministic(sp)
    else
        sp = JointTronState(p1p, p2p, h1p, h2p, trail1p, trail2p, false, 0)
        return Deterministic(sp)
    end
end

function MarkovGames.reward(game::TronMG, s::JointTronState, a, sp::JointTronState)
    r = if isterminal(game, s)
        0.0
    elseif sp.outcome == 1
        game.win_reward
    elseif sp.outcome == -1
        -game.win_reward
    else
        game.draw_reward
    end
    return SA[Float64(r), -Float64(r)]
end

"""
State encoding for neural networks.

Layout:
    [ p1_x, p1_y, h1_dx, h1_dy,
      p2_x, p2_y, h2_dx, h2_dy,
      trail1[1], ..., trail1[N],
      trail2[1], ..., trail2[N] ]

For a 10x10 board, this has dimension 8 + 2*100 = 208.
"""
function MarkovGames.convert_s(::Type{Vector{T}}, s::JointTronState, game::TronMG) where {T}
    N = n_cells(game)
    x = Vector{T}(undef, 8 + 2N)

    x[1] = T((s.p1[1] - (game.width + 1) / 2) / game.width)
    x[2] = T((s.p1[2] - (game.height + 1) / 2) / game.height)
    x[3] = T(DIRS[s.h1][1])
    x[4] = T(DIRS[s.h1][2])

    x[5] = T((s.p2[1] - (game.width + 1) / 2) / game.width)
    x[6] = T((s.p2[2] - (game.height + 1) / 2) / game.height)
    x[7] = T(DIRS[s.h2][1])
    x[8] = T(DIRS[s.h2][2])

    @inbounds for i in 1:N
        bit = UInt128(1) << (i - 1)
        x[8 + i] = T(!iszero(s.trail1 & bit))
        x[8 + N + i] = T(!iszero(s.trail2 & bit))
    end

    return x
end

# -----------------------------
# Visualization helpers
# -----------------------------

function occupied_cells(game::TronMG, trail::UInt128)
    xs = Int[]
    ys = Int[]
    for i in 1:n_cells(game)
        bit = UInt128(1) << (i - 1)
        if !iszero(trail & bit)
            p = cell_from_idx(game, i)
            push!(xs, p[1])
            push!(ys, p[2])
        end
    end
    return xs, ys
end

function action_targets(game::TronMG, s::JointTronState, player::Int)
    p = player == 1 ? s.p1 : s.p2
    h = player == 1 ? s.h1 : s.h2
    A = game.actions[player]
    return map(A) do δ
        hp = turn_heading(h, δ)
        pp = step_forward(p, hp)
        pp[1], pp[2]
    end
end

# -----------------------------
# Recipes
# -----------------------------

@recipe function f(game::TronMG, s::JointTronState)
    xlims --> (0.5, game.width + 0.5)
    ylims --> (0.5, game.height + 0.5)
    xticks --> 1:game.width
    yticks --> 1:game.height
    aspect_ratio --> 1
    legend --> false
    grid --> true
    framestyle --> :box

    xs1, ys1 = occupied_cells(game, s.trail1)
    xs2, ys2 = occupied_cells(game, s.trail2)

    @series begin
        seriestype := :scatter
        markershape := :square
        markersize := 14
        c --> :blue
        labels --> nothing
        xs1, ys1
    end

    @series begin
        seriestype := :scatter
        markershape := :square
        markersize := 14
        c --> :red
        labels --> nothing
        xs2, ys2
    end

    @series begin
        seriestype := :scatter
        markershape := :circle
        markersize := 9
        markerstrokewidth := 2
        c --> [:white, :white]
        markerstrokecolor --> [:blue, :red]
        labels --> nothing
        [s.p1[1], s.p2[1]], [s.p1[2], s.p2[2]]
    end
end

@recipe function f(game::TronMG, s::JointTronState, σ1::AbstractVector, σ2::AbstractVector)
    pts1 = action_targets(game, s, 1)
    pts2 = action_targets(game, s, 2)

    @series begin
        seriestype := :scatter
        markershape := :utriangle
        markersize := 9
        alpha --> σ1
        c --> :blue
        labels --> nothing
        first.(pts1), last.(pts1)
    end

    @series begin
        seriestype := :scatter
        markershape := :utriangle
        markersize := 9
        alpha --> σ2
        c --> :red
        labels --> nothing
        first.(pts2), last.(pts2)
    end

    @series begin
        game, s
    end
end

@recipe f(game::TronMG, s::JointTronState, σ1::SparseCat, σ2::SparseCat) = game, s, σ1.probs, σ2.probs

@recipe f(game::TronMG, s::JointTronState, σ::ProductDistribution) = game, s, σ[1], σ[2]

end
