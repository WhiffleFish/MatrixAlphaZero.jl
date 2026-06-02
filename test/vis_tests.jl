using D3Trees

@testset "vis.jl" begin
    @test AZ._vis_fmt(1.23456) == "1.235"
    @test occursin("123", AZ._vis_fmt(123.456))
    @test occursin("#d9d9d9", AZ._value_style(NaN, 1.0))
    @test occursin("#2f6b2f", AZ._value_style(0.5, 1.0))
    @test occursin("#7a2f2f", AZ._value_style(-0.5, 1.0))

    game = Fixtures.ScalarMatrixGame()
    oracle = Fixtures.TableOracle(
        values=Dict(1f0 => 0f0),
        regrets=Dict(0f0 => (Float32[0.8, 0.2], Float32[0.1, 0.9])),
        strategies=Dict(0f0 => (Float32[0.8, 0.2], Float32[0.1, 0.9])),
    )
    params = AZ.SMOOSParams(oos_iterations=1, τ=1.0, max_depth=2, oracle=oracle)
    (_, _), info = AZ.fitted_smoos_info(params, game, false; ϵ=0.0)
    tree = info.tree

    d3 = D3Trees.D3Tree(tree; title="Test Tree")
    @test length(d3.children) == length(tree.s)
    @test d3.title == "Test Tree"
    @test occursin("s1", d3.text[1])
    @test occursin("expanded = true", d3.tooltip[1])
    @test any(!isempty, d3.link_style[2:end])
end
