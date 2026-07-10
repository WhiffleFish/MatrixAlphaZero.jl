# --- SNR-SDA heuristics & outcome tracking -------------------------------------

# Games with an SDA-style layout: a satellite state vector whose first half is
# position and second half is velocity (6D `SDAState` for `SNRSDAGame`, 4D
# `SDAState2D` for `SNRGameSimple`), plus an `altitude_bounds` field.
const SDAGame = Union{SNRSDAGame, SNRGameSimple}

function _sda_altitude(x::AbstractVector)
    n = length(x) ÷ 2   # position occupies the first half of the state vector
    return norm(@view(x[1:n]), 2) - R_EARTH
end

"""
    sda_no_burn_heuristic(game, player)

Baseline policy that never burns: it always selects the zero delta-v action, so the
satellite simply coasts on its current orbit. Works for either player (observer = 1,
target = 2), and for either `SNRSDAGame` or `SNRGameSimple`.
"""
function sda_no_burn_heuristic(game::SDAGame, player::Int)
    A = actions(game)[player]
    idx = argmin(abs.(A))              # action closest to zero delta-v (no burn)
    probs = zeros(Float64, length(A))
    probs[idx] = 1.0
    return FunctionPlayerPolicy(game, player, (_g, _s) -> probs)
end

"""
    sda_no_burn_joint_policy(game)

Joint policy in which both the observer and the target maintain orbit (no burns).
"""
function sda_no_burn_joint_policy(game::SDAGame)
    return JointPolicy(sda_no_burn_heuristic(game, 1), sda_no_burn_heuristic(game, 2))
end

"""
    SDAOutcome()

Rollout accumulator for the SNR-SDA game. Tracks, per episode, whether the observer
ever detected the target (per-step SNR reward above the detection threshold while
both satellites stayed in bounds), whether the target escaped the altitude bounds,
and whether the observer left the altitude bounds.
"""
mutable struct SDAOutcome
    detected::Bool
    target_escaped::Bool
    observer_lost::Bool
end

SDAOutcome() = SDAOutcome(false, false, false)

function MarkovGames.reset!(stat::SDAOutcome)
    stat.detected = false
    stat.target_escaped = false
    stat.observer_lost = false
    return stat
end

function MarkovGames.observe_step!(stat::SDAOutcome, game::MG, step)
    r = AZ.zs_reward_scalar(step.r)
    sp = step.sp
    lo, hi = game.altitude_bounds
    observer_oob = !(lo ≤ _sda_altitude(sp.observer) ≤ hi)
    target_oob = !(lo ≤ _sda_altitude(sp.target) ≤ hi)
    stat.target_escaped |= target_oob
    stat.observer_lost |= observer_oob
    # Reward r ≥ 1 means SNR ≥ the detection threshold; only count it as a detection
    # when neither satellite is out of bounds (out-of-bounds adds the ±10 terminal
    # bonus/penalty, which would otherwise masquerade as a detection).
    if !observer_oob && !target_oob && r ≥ 1
        stat.detected = true
    end
    return stat
end

function MarkovGames.stat_result(stat::SDAOutcome)
    return (;
        detected = stat.detected,
        target_escaped = stat.target_escaped,
        observer_lost = stat.observer_lost,
    )
end

# --- Oracle value visualization ------------------------------------------------

struct OracleInfo
    obs_x::Vector{Float64}
    obs_y::Vector{Float64}
    s_target::Vector{Vector{Float64}}
    vals::Vector{Vector{Float64}}
    valrange::NTuple{2, Float64}
    sun_pos::NTuple{2, Vector{Float64}}
end

Base.eachindex(info::OracleInfo) = eachindex(info.s_target)

function OracleInfo(game, oracle, d_observer, θs = LinRange(0, 2π, 100); r_target = 5e6, n=1000)
    s_observers = rand(d_observer, n)
    oracle_info = map(θs) do θ
        s_target = sOSCtoCART([
            R_EARTH + r_target,
            0.0,
            0.0,
            0.0,
            0.0,
            θ
        ])
        s_states = map(s_observers) do s_o
            SNRGame.SDAState(s_o, s_target, game.epc0, false)
        end
        states_batch = mapreduce(hcat, s_states) do s
            MarkovGames.convert_s(Vector{Float32}, s, game)
        end
        oracle_values = Float64.(AZ.value(oracle, states_batch))
        return (;s_target, oracle_values)
    end
    vals = getfield.(oracle_info, :oracle_values)
    s_target = getfield.(oracle_info, :s_target)
    min_val, max_val = minimum(minimum.(vals)), maximum(maximum.(vals))
    # xl = xlims(p)
    # yl = ylims(p)
    # xrange = xl[2] - xl[1]
    # yrange = yl[2] - yl[1]
    sun_vec = sun_position(game.epc0)
    v̂s = normalize(sun_vec[1:2], 2)
    v0 = [0,0]
    vf = v0 .+ v̂s # .* xrange .* 0.1
    obs_x = getindex.(s_observers, 1)
    obs_y = getindex.(s_observers, 2)
    return OracleInfo(
        obs_x, obs_y, s_target, vals, (min_val, max_val), (v0, vf)
    )
end

struct OracleInfoFrame
    obs_x::Vector{Float64}
    obs_y::Vector{Float64}
    s_target::Vector{Float64}
    vals::Vector{Float64}
    valrange::NTuple{2, Float64}
    sun_pos::NTuple{2, Vector{Float64}}
end

Base.getindex(info::OracleInfo, i::Int) = OracleInfoFrame(
    info.obs_x,
    info.obs_y,
    info.s_target[i],
    info.vals[i],
    info.valrange,
    info.sun_pos
)

@recipe function f(info::OracleInfoFrame; lim=1.5e7)
    (; sun_pos) = info
    aspect_ratio --> 1.0
    size --> (800, 800)
    xticks --> [-lim, 0, lim]
    yticks --> [-lim, 0, lim]
    page_width = 2lim

    @series begin
        clims --> info.valrange,
        ms --> 10
        alpha --> 0.5
        
        seriestype := :scatter
        markerstrokewidth := 0

        zcolor := info.vals
        info.obs_x, info.obs_y
    end
    @series begin
        seriestype := :scatter
        ms --> 10
        label --> "Target"
        [info.s_target[1]], [info.s_target[2]]
    end

    @series begin
        arrow := true
        label --> "Sun Direction"
        lw --> 5
        [sun_pos[1][1], sun_pos[2][1]] .* 0.1 * page_width, [sun_pos[1][2], sun_pos[2][2]] .* 0.1 * page_width
    end
end
