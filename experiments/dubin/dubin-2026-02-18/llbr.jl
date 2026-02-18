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
    using LaTeXStrings
    using POSGModels.StaticArrays
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

game = DubinMG(V=(1.0,1.0))
# oracle = @model(50)
oracle = @model(100)

planner = AlphaZeroPlanner(game, oracle, max_iter=1000, c=10.0)
# s0 = rand(initialstate(game))
s0 = JointDubinState(SA[1,1,deg2rad(45)], SA[8,7,deg2rad(180)])

br_vals = map(readdir(@modeldir; join=true)) do modelpath
    Flux.loadmodel!(planner, modelpath)
    π1 = Tools.policy1_from_oracle(planner.oracle)
    π2 = Tools.policy2_from_oracle(planner.oracle)
    Tools.approx_br_values_both_st(game, planner.oracle, π1, π2, s0)
end

plot(
    plot(
        plot(getindex.(br_vals, 1), ylabel=L"U^1(\pi^1, \textbf{BR}(\pi^1))", c=1),
        plot(getindex.(br_vals, 2), xlabel="Training Iteration", ylabel=L"U^2(\textbf{BR}(\pi^2), \pi^2)", c=2),
        layout=(2,1)
    ),
    plot(-(getindex.(br_vals, 1) .+ getindex.(br_vals, 2)), xlabel="Training Iteration", ylabel="Exploitability", c=3),
    suptitle = "Dubin Policy Network Performance",
    layout = (1,2),
    lw=2
)

savefig(@figdir("dubin-policy-network-performance.pdf"))

init_func(::typeof(max)) = -Inf
init_func(::typeof(min)) = Inf

function running(op, v; init=init_func(op))
    val = init
    results = similar(v)
    for (i, x) in enumerate(v)
        val = op(val, x)
        results[i] = val
    end
    return results
end

plot(
    plot(
        plot!(
            plot(running(max, getindex.(br_vals, 1)), ylabel=L"U^1(\pi^1, \textbf{BR}(\pi^1))", c=1),
            getindex.(br_vals, 1), c=1, alpha=0.25
        ),
        plot!(
            plot(running(max, getindex.(br_vals, 2)), xlabel="Training Iteration", ylabel=L"U^2(\textbf{BR}(\pi^2), \pi^2)", c=2),
            getindex.(br_vals, 2), c=2, alpha=0.25
        ),
        layout=(2,1)
    ),
    plot!(
        plot(running(min, -(getindex.(br_vals, 1) .+ getindex.(br_vals, 2))), xlabel="Training Iteration", ylabel="Exploitability", c=3),
        -(getindex.(br_vals, 1) .+ getindex.(br_vals, 2)), c=3, alpha=0.25
    ),
    suptitle = "Dubin Policy Network Performance",
    layout = (1,2),
    lw=2
)

savefig(@figdir("dubin-policy-network-performance-running-max.pdf"))
