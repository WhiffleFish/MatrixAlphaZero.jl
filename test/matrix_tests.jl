using MarkovGames

@testset "matrix.jl" begin
    x = fill(-1.0, 3)
    AZ.match!(x, [-2.0, 3.0, 0.0])
    @test x ≈ [0.0, 1.0, 0.0]

    y = zeros(4)
    AZ.match!(y, fill(-1.0, 4))
    @test y ≈ fill(0.25, 4)

    A = [
        1.0 -1.0
        -1.0 1.0
    ]
    x0, y0, v0 = MarkovGames.solve(A)
    @test x0 ≈ [0.5, 0.5] atol = 1e-3
    @test y0 ≈ [0.5, 0.5] atol = 1e-3
    @test isapprox(v0, 0.0; atol=1e-6)

    xr, yr, vr = AZ.regret_match(A'; n=100)
    @test xr ≈ [0.5, 0.5] atol = 0.05
    @test yr ≈ [0.5, 0.5] atol = 0.05
    @test isapprox(vr, 0.0; atol=0.05)

    sol = AZ.RegretSolver(100)
    x1, y1, v1 = MarkovGames.solve(sol, A)
    @test x1 ≈ [0.5, 0.5] atol = 0.05
    @test y1 ≈ [0.5, 0.5] atol = 0.05
    @test isapprox(v1, 0.0; atol=0.05)

    x2, y2, v2 = MarkovGames.solve(AZ.PATHSolver(), A)
    @test x2 ≈ [0.5, 0.5] atol = 1e-6
    @test y2 ≈ [0.5, 0.5] atol = 1e-6
    @test isapprox(v2, 0.0; atol=1e-6)

    B = -A
    x3, y3, v3 = MarkovGames.solve(sol, A, B)
    @test x3 ≈ [0.5, 0.5] atol = 0.05
    @test y3 ≈ [0.5, 0.5] atol = 0.05
    @test isapprox(v3, 0.0; atol=0.05)
end
