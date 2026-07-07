module Tron

using MarkovGames
using MatrixAlphaZero

const AZ = MatrixAlphaZero
const Tools = parentmodule(@__MODULE__)

export TronOutcome
export tron_heuristic, tron_heuristic_joint_policy

# -----------------------------
# Grid helpers (mirror POSGModels.Tron internals; kept local so we don't
# depend on that package's unexported utilities)
# -----------------------------

# Heading convention: 1 => up, 2 => right, 3 => down, 4 => left
const DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))

_turn(h::Int, δ::Int) = mod1(h + δ, 4)

_inbounds(game, x::Int, y::Int) = 1 <= x <= game.width && 1 <= y <= game.height

_linidx(game, x::Int, y::Int) = x + (y - 1) * game.width

_cellbit(game, x::Int, y::Int) = UInt128(1) << (_linidx(game, x, y) - 1)

_occupied(trails::UInt128, bit::UInt128) = !iszero(trails & bit)

"""
    _bfs_dist(game, blocked, sources)

Breadth-first flood over the free cells of the board. `blocked` is a bitboard of
occupied cells; `sources` is an iterable of `(x, y)` seed cells (each assigned
distance 0 even if their own bit is set in `blocked`, since a player's head sits
on an occupied cell). Returns a `Vector{Int}` of length `width*height` giving the
shortest free-cell distance to each linear index, or `typemax(Int)` if unreachable.
"""
function _bfs_dist(game, blocked::UInt128, sources)
    N = game.width * game.height
    dist = fill(typemax(Int), N)
    queue = Int[]
    for (sx, sy) in sources
        _inbounds(game, sx, sy) || continue
        idx = _linidx(game, sx, sy)
        if dist[idx] == typemax(Int)
            dist[idx] = 0
            push!(queue, idx)
        end
    end
    head = 1
    while head <= length(queue)
        idx = queue[head]
        head += 1
        d = dist[idx]
        x = 1 + (idx - 1) % game.width
        y = 1 + (idx - 1) ÷ game.width
        for (dx, dy) in DIRS
            nx = x + dx
            ny = y + dy
            _inbounds(game, nx, ny) || continue
            nbit = _cellbit(game, nx, ny)
            _occupied(blocked, nbit) && continue
            nidx = _linidx(game, nx, ny)
            if dist[nidx] == typemax(Int)
                dist[nidx] = d + 1
                push!(queue, nidx)
            end
        end
    end
    return dist
end

"""
    _voronoi_scores(game, blocked, me, opp)

Given a bitboard `blocked` of occupied cells and the two heads `me`/`opp` as
`(x, y)` tuples, flood from each head over the free space and partition cells by
who reaches them first (ties go to neither). Returns `(mine, theirs, reach)`:
the number of cells I reach strictly first, the number the opponent reaches
strictly first, and the total number of free cells I can reach at all (my
breathing room / survival space).
"""
function _voronoi_scores(game, blocked::UInt128, me, opp)
    N = game.width * game.height
    mydist = _bfs_dist(game, blocked, (me,))
    oppdist = _bfs_dist(game, blocked, (opp,))
    mine = 0
    theirs = 0
    reach = 0
    @inbounds for i in 1:N
        md = mydist[i]
        od = oppdist[i]
        if md < typemax(Int)
            reach += 1
        end
        if md < od
            mine += 1
        elseif od < md
            theirs += 1
        end
    end
    return mine, theirs, reach
end

"""
    tron_heuristic(game, player; opp_weight=1.0)

A reasonably strong hand-coded baseline for `TronMG`, built as a
`FunctionPlayerPolicy`. Both players are symmetric light-cycles, so the same
routine serves either `player`.

Strategy (a classic space-control / Voronoi bot):

1. **Survival filter** — never take a relative turn whose next cell is out of
   bounds or already part of either trail (an immediate crash).
2. **Territory maximization** — among the surviving moves, flood-fill from both
   heads over the remaining free space and pick the move that maximizes the
   Voronoi differential `|cells I reach first| - opp_weight * |cells opp reaches
   first|`. This is the dominant heuristic in competitive Tron AI: it both grabs
   open area and races the opponent for contested territory.
3. **Tie-breaks** — prefer the move that keeps the most reachable free space
   (avoid sealing yourself into a small pocket), then prefer going straight for
   smoother, wall-hugging paths.

Returns a deterministic one-hot distribution over the player's action indices
(matching the style of the Dubin heuristic). `opp_weight` trades off aggression
(chasing/cutting off the opponent) against pure space-filling.
"""
function tron_heuristic(game::MG, player::Int; opp_weight::Float64=1.0)
    return Tools.FunctionPlayerPolicy(game, player) do game, s
        A = game.actions[player]
        n = length(A)

        p = isone(player) ? s.p1 : s.p2
        h = isone(player) ? s.h1 : s.h2
        opp = isone(player) ? s.p2 : s.p1
        px, py = Int(p[1]), Int(p[2])
        opp_cell = (Int(opp[1]), Int(opp[2]))
        occ = s.trail1 | s.trail2

        best = -Inf
        best_reach = -1
        best_straight = false
        best_i = 0

        for i in 1:n
            δ = A[i]
            hp = _turn(h, δ)
            dx, dy = DIRS[hp]
            nx = px + dx
            ny = py + dy

            # survival filter: reject immediate crashes
            (_inbounds(game, nx, ny) && !_occupied(occ, _cellbit(game, nx, ny))) || continue

            blocked = occ | _cellbit(game, nx, ny)
            mine, theirs, reach = _voronoi_scores(game, blocked, (nx, ny), opp_cell)
            score = mine - opp_weight * theirs
            straight = δ == 0

            better = score > best ||
                     (score == best && reach > best_reach) ||
                     (score == best && reach == best_reach && straight && !best_straight)
            if better
                best = score
                best_reach = reach
                best_straight = straight
                best_i = i
            end
        end

        probs = zeros(Float64, n)
        # if every move is fatal, go straight if possible, else first action
        if iszero(best_i)
            straight_i = findfirst(==(0), A)
            best_i = isnothing(straight_i) ? 1 : straight_i
        end
        probs[best_i] = 1.0
        return probs
    end
end

"""
    tron_heuristic_joint_policy(game; kwargs...)

Both players controlled by [`tron_heuristic`](@ref). `kwargs` are forwarded to
each player's heuristic.
"""
function tron_heuristic_joint_policy(game::MG; kwargs...)
    return Tools.JointPolicy(
        tron_heuristic(game, 1; kwargs...),
        tron_heuristic(game, 2; kwargs...),
    )
end

# -----------------------------
# Outcome statistics accumulator (parallels DubinOutcome)
# -----------------------------

mutable struct TronOutcome
    p1_win::Bool
    p2_win::Bool
    draw::Bool
end

TronOutcome() = TronOutcome(false, false, false)

function MarkovGames.reset!(stat::TronOutcome)
    stat.p1_win = false
    stat.p2_win = false
    stat.draw = false
    return stat
end

function MarkovGames.observe_step!(stat::TronOutcome, game::MG, step)
    r = AZ.zs_reward_scalar(step.r)
    stat.p1_win |= r > 0
    stat.p2_win |= r < 0
    # a draw is a terminal transition (both crash) with zero reward
    if step.sp.terminal && r == 0
        stat.draw = true
    end
    return stat
end

function MarkovGames.stat_result(stat::TronOutcome)
    return (;
        p1_win = stat.p1_win,
        p2_win = stat.p2_win,
        draw = stat.draw,
        timeout = !(stat.p1_win || stat.p2_win || stat.draw),
    )
end

end # module Tron
