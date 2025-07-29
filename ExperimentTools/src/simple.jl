using POSGModels.SimpleGame

struct SimpleOracle{D<:Dict} <: Policy
    d::D
end

function MarkovGames.behavior(oracle::SimpleOracle, s)
    na1, na2 = length.(first(values(oracle.d)).policy)
    res = get(oracle.d, s, nothing)
    if isnothing(res)
        return ProductDistribution(SparseCat(1:na1, AZ.uniform(na1)), SparseCat(1:na2, AZ.uniform(na2)))
    else # stored entry
        p1, p2 = res.policy[1], res.policy[2]
        return ProductDistribution(SparseCat(1:na1, p1), SparseCat(1:na2, p2))
    end
end

SimpleOracle(game::SimpleMG; solver=PATHSolver()) = SimpleOracle(SimpleGame.exact_solve(solver, game))

function AZ.state_value(oracle::SimpleOracle, game, s)
    haskey(oracle.d, s) ? oracle.d[s].value : 0.0
end
AZ.batch_state_value(oracle::SimpleOracle, game, sv) = map(sv) do s_i
    AZ.state_value(oracle, game, s_i)
end
AZ.state_policy(oracle::SimpleOracle, game, s) = haskey(oracle.d, s) ? oracle.d[s].policy : (uniform.(length.(actions(game))))
