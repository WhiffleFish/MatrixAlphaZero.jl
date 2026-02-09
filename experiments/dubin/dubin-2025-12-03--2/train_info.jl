begin
    using Pkg
    Pkg.activate("experiments")
    using JLD2
    using Plots
    using ExperimentTools
    default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
end

info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
plot(
    plot(reduce(vcat, info["train_losses"]), title="Total Loss"),
    plot(
        plot(reduce(vcat, info["value_losses"]), title="Value Loss"), 
        plot(reduce(vcat, info["policy_losses"]), title="Policy Loss")
    ),
    layout = (2,1)
)
savefig(@figdir("train_info.svg"))
savefig(@figdir("train_info.pdf"))


##
using MarkovGames
using POSGModels.Dubin
using MatrixAlphaZero
const AZ = MatrixAlphaZero
using Flux

buf = info["buffer"]
game = DubinMG(V = (1.0, 1.0))
# trunk = Chain(Dense(8, 16, tanh), Dense(16, 16, tanh))
trunk = identity
critic = AZ.HLGaussCritic(
    Chain(Dense(8, 32, gelu), Dense(32, 32, gelu), Dense(32, 64)),
    -20, 20, 64
)
na1, na2 = length.(actions(game))
actor = MultiActor(
    Chain(Dense(8, 32, gelu), Dense(32, 32, gelu), Dense(32, na1)), 
    Chain(Dense(8, 32, gelu), Dense(32, 32, gelu), Dense(32, na2))
)
oracle = ActorCritic(trunk, actor, critic)
optimiser = Flux.Optimisers.OptimiserChain(
    # Flux.Optimisers.ClipNorm(1f0),
    # Flux.Optimisers.ClipGrad(1f0),
    Flux.Optimisers.Adam(1f-2)
)
opt_state = Flux.setup(optimiser, oracle)
X = reduce(hcat, buf.s)
v_target = Array(buf.v)
v_target = AZ.prepare_target(oracle.critic, v_target)
p_target = reduce(hcat, buf.p[1]), reduce(hcat, buf.p[2])
losses = Float32[]
value_losses = Float32[]
policy_losses = Float32[]
n_batches = 1000
λ = 1f-4
for i ∈ 1:n_batches
    @show i
    ∇θ = Flux.gradient(oracle) do oracle
        lv, lp = AZ.loss(oracle, X, v_target, p_target)
        l = lv + lp + λ * sum(pen_l2, Flux.trainables(oracle))
        Flux.Zygote.ignore_derivatives() do
            push!(losses, l)
            push!(value_losses, lv)
            push!(policy_losses, lp)
        end
        return l
    end
    Flux.update!(opt_state, oracle, ∇θ[1])
end

plot(losses)
plot(policy_losses)
plot(value_losses)

pen_l2(x::AbstractArray) = sum(abs2, x) / 2
sum(pen_l2, Flux.trainables(oracle))

