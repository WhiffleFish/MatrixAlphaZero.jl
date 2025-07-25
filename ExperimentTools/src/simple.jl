using POSGModels.SimpleGame

struct SimpleOracle{D<:Dict} <: Policy
    d::D
end

SimpleOracle(game::SimpleMG) = SimpleOracle(SimpleGame.exact_solve(RegretSolver(20), game))

function AZ.state_value(oracle::SimpleOracle, game, s)
    haskey(oracle.d, s) ? oracle.d[s].value : 0.0
end
AZ.batch_state_value(oracle::SimpleOracle, game, sv) = map(sv) do s_i
    AZ.state_value(oracle, game, s_i)
end
AZ.state_policy(oracle::SimpleOracle, game, s) = haskey(oracle.d, s) ? oracle.d[s].policy : (uniform.(length.(actions(game))))
