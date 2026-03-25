using JLD2

@testset "callbacks.jl" begin
    seen = Int[]
    AZ.call((
        info -> push!(seen, info.iter),
        info -> push!(seen, info.iter + 10),
    ), (; iter=2))
    @test seen == [2, 12]

    @test AZ.iter2string(7) == "0007"
    @test AZ.iter2string(12, 2) == "12"

    mktempdir() do dir
        model = Fixtures.simple_actor_critic()
        cb = AZ.ModelSaveCallback(dir)

        cb((; oracle=model))
        cb((; oracle=model, iter=3))

        @test isfile(joinpath(dir, "oracle.jld2"))
        @test isfile(joinpath(dir, "oracle0003.jld2"))
        @test haskey(JLD2.load(joinpath(dir, "oracle0003.jld2")), "model_state")
    end
end
