begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using POSGModels.Dubin
    using Flux
    using POMDPTools
    using POMDPs
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

game = DubinMG()
oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
planner = AlphaZeroPlanner(game, oracle, max_iter=10, c=10.0)
s0 = rand(initialstate(game))
s0 = JointDubinState([3,3,0], [6,6,π])

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)  # or your own policy function
    Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
end

plot(br_vals, xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")

x,y,t = solve(planner.matrix_solver, AZ.oracle_matrix_game(game, Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0000.jld2")), s0))
x
AZ.state_policy(
    Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0000.jld2")), 
    game, 
    s0
)

function policy1_from_value_oracle(oracle, matrix_solver)
    return function (game, s)
        x,y,t = solve(matrix_solver, AZ.oracle_matrix_game(game, oracle, s))
        return x
    end
end

function policy1_from_planner(planner::AlphaZeroPlanner)
    return function (game, s)
        σ = behavior(planner, s)
        return σ[1].probs
    end
end

br_vals2 = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = policy1_from_value_oracle(planner.oracle, planner.matrix_solver)
    Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
end

plot(br_vals2, xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")


planner = AlphaZeroPlanner(game, oracle, max_iter=100, c=10.0)
br_vals3 = map(readdir(@modeldir; join=true)) do modelpath
    @show modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = policy1_from_planner(planner)
    Tools.approx_br_value(game, planner.oracle, π1, s0; max_depth=5)
end

plot(br_vals3, xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")

