module Dubin

using MarkovGames
using MatrixAlphaZero

const AZ = MatrixAlphaZero

export DubinOutcome

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
    reward = AZ.zs_reward_scalar(step.r)
    stat.attacker_goal |= reward > 0
    stat.tagged |= reward < 0
    return stat
end

function MarkovGames.stat_result(stat::DubinOutcome)
    return (;
        attacker_goal=stat.attacker_goal,
        tagged=stat.tagged,
        timeout=!(stat.attacker_goal || stat.tagged),
    )
end

end
