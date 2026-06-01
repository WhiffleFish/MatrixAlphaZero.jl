module Dubin

using LinearAlgebra
using MarkovGames
using MatrixAlphaZero

const AZ = MatrixAlphaZero
const Tools = parentmodule(@__MODULE__)

export DubinOutcome
export dubin_attacker_heuristic, dubin_defender_heuristic, dubin_heuristic_joint_policy

mutable struct DubinOutcome
    attacker_goal::Bool
    tagged::Bool
end

DubinOutcome() = DubinOutcome(false, false)

function MarkovGames.reset!(stat::DubinOutcome)
    stat.attacker_goal = false
    stat.tagged = false
    return stat
end

function MarkovGames.observe_step!(stat::DubinOutcome, game::MG, step)
    r = AZ.zs_reward_scalar(step.r)
    stat.attacker_goal |= r > 0
    stat.tagged |= r < 0
    return stat
end

function MarkovGames.stat_result(stat::DubinOutcome)
    return (;
        attacker_goal = stat.attacker_goal,
        tagged = stat.tagged,
        timeout = !(stat.attacker_goal || stat.tagged),
    )
end

function _dubin_heading_vec(θ)
    return [cos(θ), sin(θ)]
end

function _dubin_unit(v; fallback=[1.0, 0.0])
    n = norm(v)
    return n <= eps(Float64) ? fallback : v ./ n
end

function _dubin_onehot_turn_policy(game, player::Int, state, desired)
    A = collect(actions(game)[player])
    rates = game.actions[player]
    agent = isone(player) ? state.attacker : state.defender
    desired_unit = _dubin_unit(desired; fallback=_dubin_heading_vec(agent[3]))
    scores = map(A) do action_idx
        θ = agent[3] + rates[action_idx] * game.dt
        h = _dubin_heading_vec(θ)
        return dot(h, desired_unit)
    end
    probs = zeros(Float64, length(A))
    probs[argmax(scores)] = 1.0
    return probs
end

function dubin_attacker_heuristic(
        game::MG;
        evade_radius::Float64=2.0,
        evade_weight::Float64=1.0,
        goal_weight::Float64=1.0,
    )
    return Tools.FunctionPlayerPolicy(game, 1) do game, s
        attacker = s.attacker[1:2]
        defender = s.defender[1:2]
        goal = game.goal.center
        to_goal = goal_weight .* _dubin_unit(goal .- attacker; fallback=_dubin_heading_vec(s.attacker[3]))
        away = attacker .- defender
        dist = norm(away)
        desired = to_goal
        if 0 < dist < evade_radius
            evade = evade_weight * (evade_radius - dist) / evade_radius .* _dubin_unit(away)
            desired = desired .+ evade
        end
        return _dubin_onehot_turn_policy(game, 1, s, desired)
    end
end

function dubin_defender_heuristic(game::MG)
    return Tools.FunctionPlayerPolicy(game, 2) do game, s
        attacker = s.attacker[1:2]
        defender = s.defender[1:2]
        desired = attacker .- defender
        return _dubin_onehot_turn_policy(game, 2, s, desired)
    end
end

function dubin_heuristic_joint_policy(game::MG; attacker_kwargs...)
    return Tools.JointPolicy(dubin_attacker_heuristic(game; attacker_kwargs...), dubin_defender_heuristic(game))
end

end # module Dubin
