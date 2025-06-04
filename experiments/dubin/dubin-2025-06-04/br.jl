begin
    using MatrixAlphaZero
    const AZ = MatrixAlphaZero
    using Plots
    using JLD2
    using Flux
    using MarkovGames
    using POMDPs
    using POMDPTools
    using POSGModels.Dubin
    using MCTS
    using Plots
    using ProgressMeter
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

b0 = ImplicitDistribution() do rng
    s1 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    s2 = Dubin.Vec3(rand(rng) * 10, rand(rng) * 10, rand(rng) * 2π)
    return JointDubinState(s1, s2)
end
game = DubinMG()

init = Flux.orthogonal(; gain=sqrt(2))
trunk = Chain(Dense(8, 32, tanh; init), Dense(32, 32, tanh; init))
critic = AZ.HLGaussCritic(
    Chain(Dense(32, 32, tanh; init), Dense(32, 32; init)),
    -15, 15, 32
)
actor = MultiActor(
    Chain(Dense(32, 16, tanh; init), Dense(16, 3; init)), 
    Chain(Dense(32, 16, tanh; init), Dense(16, 3; init))
)

oracle = ActorCritic(trunk, actor, critic)
mdp_sol = MCTSSolver(n_iterations=1000)

s = JointDubinState(Dubin.Vec3(0,0,π/4), Dubin.Vec3(5,5,-3π/4))

model_nums = eachindex(readdir(@modeldir)) .- 1
br_vals_1 = @showprogress map(model_nums) do i
    model_state = JLD2.load(joinpath(@__DIR__, "models", "oracle"*AZ.iter2string(i)*".jld2"), "model_state")
    Flux.loadmodel!(oracle, model_state)
    pol = AlphaZeroPlanner(oracle, game, max_iter=0)
    mdp = ExploitabilityMDP(game, pol, 1)
    mdp_pol = solve(mdp_sol, mdp)
    a, mdp_a_info = action_info(mdp_pol, s)
    maximum(mdp_a_info.tree.q[1:length(actions(game)[1])])
end

plot(
    model_nums, 
    -br_vals_1, 
    lw=2, 
    xlabel="AlphaZero Iteration", 
    ylabel="Exploiter BR value", 
    xticks=model_nums, 
    title="Dubin Evaluation"
)
savefig(joinpath(@__DIR__, "figures", "dubin_p1_brv.pdf"))
savefig(joinpath(@__DIR__, "figures", "dubin_p1_brv.png"))

##

# s = JointDubinState(Dubin.Vec3(0,0,π/4), Dubin.Vec3(5,5,-3π/4))
prog = Progress(length(model_nums))
br_vals_2 = map(model_nums) do i
    model_state = JLD2.load(joinpath(@__DIR__, "models", "oracle"*AZ.iter2string(i)*".jld2"), "model_state")
    Flux.loadmodel!(oracle, model_state)
    pol = AlphaZeroPlanner(oracle, game, max_iter=0)
    mdp = ExploitabilityMDP(game, pol, 2)
    mdp_pol = solve(mdp_sol, mdp)
    a, mdp_a_info = action_info(mdp_pol, s)
    next!(prog)
    maximum(mdp_a_info.tree.q[1:length(actions(game)[2])])
end
finish!(prog)

plot(
    model_nums, 
    -br_vals_2, 
    lw=2, 
    xlabel="AlphaZero Iteration", 
    ylabel="AlphaZero Value", 
    xticks=model_nums, 
    title="AlphaZero Pursuer vs MCTS Exploitative Evader"
)
savefig(joinpath(@__DIR__, "figures", "dubin_p2_brv.pdf"))
savefig(joinpath(@__DIR__, "figures", "dubin_p2_brv.png"))


##
plot(
    model_nums, 
    br_vals_1 + br_vals_2, 
    lw=2, 
    xlabel="AlphaZero Iteration", 
    ylabel="Exploitability", 
    xticks=model_nums, 
    title="Tag Exploitability"
)
savefig(joinpath(@__DIR__, "figures", "tag_exploitability.pdf"))
savefig(joinpath(@__DIR__, "figures", "tag_exploitability.png"))

plot(-br_vals_1)
plot(-br_vals_2)

plot(-accumulate(min, br_vals_2))
plot(-accumulate(min, br_vals_1))
plot(model_nums, accumulate(min, br_vals_1 + br_vals_2), lw=2, c=1)
plot!(model_nums, br_vals_1 + br_vals_2, lw=2, alpha=0.1, c=1)


## 19 seems the best
i = 19
model_state = JLD2.load(joinpath(@__DIR__, "models", "oracle"*AZ.iter2string(i)*".jld2"), "model_state")
Flux.loadmodel!(oracle, model_state)
pol = AlphaZeroPlanner(oracle, game, max_iter=0)


##
a = rand.(actions(game))
first(Base.return_types(@gen(:r), typeof((game, s, a))))
zero(Tuple{Float64, Float64})
