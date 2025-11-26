begin
    Pkg.activate("experiments")
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using POSGModels.Dubin
    using Flux
    using POMDPTools
    using POMDPs
    using ProgressMeter
    using Plots
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

game = DubinMG(V=(1.0,1.0))
oracle0 = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
oracle = Flux.loadmodel!(AZ.load_oracle(@__DIR__), @modeldir("oracle0100.jld2"))
planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
s0 = rand(initialstate(game))
# s0 = JointDubinState([3,3,0], [6,6,π])

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)
    π2 = Tools.policy2_from_oracle(planner.oracle)
    Tools.approx_br_values_both_st(game, oracle, π1, π2, s0; value_oracle=oracle0)
end

plot(getindex.(br_vals, 1), xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")
plot(getindex.(br_vals, 2), xlabel="Training Iteration", ylabel="BRV", lw=2, title="Dubin Tag")
