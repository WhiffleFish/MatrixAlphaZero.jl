function gen_heatmap_vals(game::TagMG, oracle, x, y)
    V = zeros(game.floor...)
    for i ∈ 1:game.floor[1], j ∈ 1:game.floor[2]
        s_vec = MarkovGames.convert_s(
            Vector{Float32}, 
            TagState(Coord(x,y), Coord(i,j), false), 
            game
        )
        V[i,j] = only(oracle(s_vec))
    end
    return V
end

@recipe function f(game::TagMG, oracle, x, y)
    seriestype := :heatmap
    aspect_ratio --> 1.0
    gen_heatmap_vals(game, oracle, x, y)'
end

function action_lines(x::Coord)
    return map(DiscreteTag.ACTION_DIRS) do a
        sp = x + a
        [x[1], sp[1]], [x[2], sp[2]]
    end
end

@recipe function f(oracle, s::TagState)
    (;pursuer, evader) = s
    x,y,t = AZ.solve(AZ.oracle_matrix_game(game, oracle, s))
    @series begin
        seriestype  := :scatter
        c           --> [1,2]
        [pursuer[1], evader[1]], [pursuer[2], evader[2]]
    end
    @series begin
        c       --> 1
        lw      --> 10
        alpha   --> x |> permutedims
        action_lines(pursuer)
    end
    @series begin
        c       --> 2
        lw      --> 10
        alpha   --> y |> permutedims
        action_lines(evader)
    end
end
