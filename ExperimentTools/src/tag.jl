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
