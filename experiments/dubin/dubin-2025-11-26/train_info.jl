using JLD2
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
info = jldopen(joinpath(@__DIR__, "train_info.jld2"))
plot(
    plot(reduce(vcat, info["train_losses"]), title="Total Loss"),
    plot(
        plot(reduce(vcat, info["value_losses"]), title="Value Loss"), 
        plot(reduce(vcat, info["policy_losses"]), title="Policy Loss")
    ),
    layout = (2,1)
)

##
using Flux
m = chain = Chain(Dense(3,16), Dense(16,2))

A = randn(2,3)
b = randn(2)
f(x) = A*x + b
X = [randn(3) for _ in 1:10]
_X = Float32.(reduce(hcat, X))
y = Float32.(reduce(hcat, [f(x) + randn(2)*0.1 for x ∈ X]))

pen_l2(x::AbstractArray) = sum(abs2, x) / 2

∇V = Flux.gradient(chain) do chain
    l = sum(abs2, chain(_X) .- y)
    return l
end

l2_penalty = sum(pen_l2, Flux.trainables(m))

