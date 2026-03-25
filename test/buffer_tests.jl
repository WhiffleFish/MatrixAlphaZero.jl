@testset "buffer.jl" begin
    buf = AZ.Buffer(3)
    hist = (
        s = [Float32[0], Float32[1]],
        v = Float32[1, 2],
        policy = (
            [Float32[1, 0], Float32[0, 1]],
            [Float32[0, 1], Float32[1, 0]],
        ),
    )
    push!(buf, hist)
    @test length(buf) == 2
    @test collect(eachindex(buf)) == [1, 2]

    push!(buf, (
        s = [Float32[2], Float32[3]],
        v = Float32[3, 4],
        policy = (
            [Float32[0.2, 0.8], Float32[0.7, 0.3]],
            [Float32[0.6, 0.4], Float32[0.1, 0.9]],
        ),
    ))
    @test length(buf) == 3
    @test buf[1].s == Float32[1]
    @test buf[length(buf)].v == 4f0

    slice = buf[1:2]
    @test length(slice.s) == 2
    @test length(slice.p) == 2
    @test all(length.(slice.p) .== 2)

    empty!(buf)
    @test length(buf) == 0
end
