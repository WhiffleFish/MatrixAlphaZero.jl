using Plots
using DelimitedFiles
using Statistics
br1 = DelimitedFiles.readdlm(joinpath(@__DIR__, "brv-c1/br1.csv"), ',')
br2 = DelimitedFiles.readdlm(joinpath(@__DIR__, "brv-c1/br2.csv"), ',')

iter = axes(br1, 1) .- 1

m1 = mean(br1, dims=2) |> vec
s1 = std(br1, dims=2) |> vec
m2 = mean(br2, dims=2) |> vec
s2 = std(br2, dims=2) |> vec

p_br1 = plot(iter, m1, ribbon=s1, ylabel="BRV1")
p_br2 = plot(iter, m2, ribbon=s2, xlabel="Training Iteration", ylabel="BRV2")

plot(p_br1, p_br2, layout=(2,1))
savefig(joinpath(@__DIR__, "figures", "BRV-c1.pdf"))
savefig(joinpath(@__DIR__, "figures", "BRV-c1.png"))

p_exp = plot(iter, m1 + m2, ribbon=sqrt.( s1 .^2 .+ s2 .^2 ), xlabel="", ylabel="exploitability")
savefig(p_exp, joinpath(@__DIR__, "figures", "exp-c1.pdf"))
savefig(p_exp, joinpath(@__DIR__, "figures", "exp-c1.png"))


br1 = DelimitedFiles.readdlm(joinpath(@__DIR__, "brv-c10/br1.csv"), ',')
br2 = DelimitedFiles.readdlm(joinpath(@__DIR__, "brv-c10/br2.csv"), ',')

iter = axes(br1, 1) .- 1

m1 = mean(br1, dims=2) |> vec
s1 = std(br1, dims=2) |> vec
m2 = mean(br2, dims=2) |> vec
s2 = std(br2, dims=2) |> vec

p_br1 = plot(iter, m1, ribbon=s1, ylabel="BRV1")
p_br2 = plot(iter, m2, ribbon=s2, xlabel="Training Iteration", ylabel="BRV2")

plot(p_br1, p_br2, layout=(2,1))
savefig(joinpath(@__DIR__, "figures", "BRV-c10.pdf"))
savefig(joinpath(@__DIR__, "figures", "BRV-c10.png"))

p_exp = plot(iter, m1 + m2, ribbon=sqrt.( s1 .^2 .+ s2 .^2 ), xlabel="", ylabel="exploitability")
savefig(p_exp, joinpath(@__DIR__, "figures", "exp-c10.pdf"))
savefig(p_exp, joinpath(@__DIR__, "figures", "exp-c10.png"))

