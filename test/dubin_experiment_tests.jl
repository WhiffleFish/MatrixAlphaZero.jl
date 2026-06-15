@testset "Dubin 2026-06-15 experiment API surface" begin
    train_path = joinpath(@__DIR__, "..", "experiments", "dubin", "dubin-2026-06-15", "train.jl")
    source = read(train_path, String)

    @test !isnothing(Meta.parse("begin\n$(source)\nend"))
    @test occursin("const EXPERIMENT_NAME = \"dubin-2026-06-15\"", source)
    @test occursin("AZ.FittedRegretModel(", source)
    @test occursin("value_weight,", source)
    @test occursin("regret_weight,", source)
    @test occursin("strategy_weight,", source)
    @test occursin("search = AZ.SMOOSSearch", source)
    @test occursin("search = search", source)
    @test occursin("\"oracle/value_weight\" => oracle.value_weight", source)
    @test occursin("\"search/transfer_weight\" => search.transfer_weight", source)
    @test !occursin("smoos_params", source)
    @test !occursin("search_params", source)
    @test !occursin("AZ.SMOOSParams", source)
end
