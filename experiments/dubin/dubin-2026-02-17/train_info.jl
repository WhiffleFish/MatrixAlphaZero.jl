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
