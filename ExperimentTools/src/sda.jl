function sda_altitude(state::AbstractVector)
    position_dimension = length(state) ÷ 2
    return sqrt(sum(abs2, @view(state[1:position_dimension]))) - R_EARTH
end

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
    reward = AZ.zs_reward_scalar(step.r)
    lower, upper = game.altitude_bounds
    observer_out = !(lower ≤ sda_altitude(step.sp.observer) ≤ upper)
    target_out = !(lower ≤ sda_altitude(step.sp.target) ≤ upper)
    stat.observer_lost |= observer_out
    stat.target_escaped |= target_out
    stat.detected |= !observer_out && !target_out && reward ≥ 1
    return stat
end

function MarkovGames.stat_result(stat::SDAOutcome)
    return (;
        detected=stat.detected,
        target_escaped=stat.target_escaped,
        observer_lost=stat.observer_lost,
    )
end
