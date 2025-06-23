struct Tree{S}
    s           ::  Vector{S}                           # [s_idx] -> s
    s_children  ::  Vector{Matrix{Int}}                 # [s_idx][a1, a2] -> sp_idx
    v_lower     ::  Vector{Matrix{Float64}}             # [s_idx][a1, a2] -> V̲(T(s, a1, a2))
    v_upper     ::  Vector{Matrix{Float64}}             # [s_idx][a1, a2] -> V̄(T(s, a1, a2))
    solved_vl   ::  Vector{Float64}
    solved_vu   ::  Vector{Float64}
    π̄           ::  NTuple{2,Vector{Vector{Float64}}}
    π̲           ::  NTuple{2,Vector{Vector{Float64}}}
    r           ::  Vector{Matrix{Float64}}             # [s_idx][a1, a2] -> r(s, a1, a2)
    γ           ::  Float64 
end

MarkovGames.discount(tree::Tree) = tree.γ

const NO_CHILDREN = Matrix{Int}(undef, 0, 0)

function Tree(game::MG, s=rand(initialstate(game)))
    return Tree(
        [s],                            # s
        Matrix{Int}[NO_CHILDREN],       # s_children
        [Matrix{Float64}(undef, 0, 0)], # Q̲
        [Matrix{Float64}(undef, 0, 0)], # Q̄
        Float64[0.0],
        Float64[0.0],
        ([Vector{Float64}(undef, 0)], [Vector{Float64}(undef, 0)]),
        ([Vector{Float64}(undef, 0)], [Vector{Float64}(undef, 0)]),
        [Matrix{Float64}(undef, 0, 0)],  # r
        discount(game)
    )
end

is_leaf(tree::Tree, s_idx::Int) = isempty(tree.s_children[s_idx])

function expand_s!(tree::Tree, s_idx::Int, game::MG, solver)
    if is_leaf(tree, s_idx)
        _expand_s!(tree, s_idx, game, solver)
    end
end

function _expand_s!(tree::Tree, s_idx::Int, game::MG, solver)
    s = tree.s[s_idx]
    A1, A2 = actions(game)
    na1, na2 = length(A1), length(A2)
    s_children = zeros(Int, na1, na2)
    r = zeros(Float64, na1, na2)
    counter = length(tree.s) + 1
    frontier = statetype(game)[]

    for (j,a2) ∈ enumerate(A2), (i,a1) ∈ enumerate(A1)
        s_children[i,j] = counter
        sp, r_i = @gen(:sp, :r)(game, s, (a1, a2))
        push!(frontier, sp)
        r[i,j] = r_i
        counter += 1
    end
    n_frontier = length(frontier)

    tree.s_children[s_idx] = s_children
    tree.v_lower[s_idx] = fill(solver.v_lower, na1, na2)
    tree.v_upper[s_idx] = fill(solver.v_upper, na1, na2)
    tree.r[s_idx] = r

    x̲, y̲, V̲ = AZ.solve(lower_bound_matrix_game(tree, s_idx))
    x̄, ȳ, V̄ = AZ.solve(upper_bound_matrix_game(tree, s_idx))

    tree.π̲[1][s_idx] = x̲
    tree.π̲[2][s_idx] = y̲
    tree.π̄[1][s_idx] = x̄
    tree.π̄[2][s_idx] = ȳ

    tree.solved_vl[s_idx] = V̲
    tree.solved_vu[s_idx] = V̄

    
    append!(tree.s, frontier)
    append!(tree.v_lower, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.v_upper, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.solved_vl, fill(solver.v_lower, n_frontier))
    append!(tree.solved_vu, fill(solver.v_upper, n_frontier))
    append!(tree.s_children, fill(NO_CHILDREN, n_frontier))
    append!(tree.π̲[1], fill(Vector{Float64}(undef, 0), n_frontier))
    append!(tree.π̲[2], fill(Vector{Float64}(undef, 0), n_frontier))
    append!(tree.π̄[1], fill(Vector{Float64}(undef, 0), n_frontier))
    append!(tree.π̄[2], fill(Vector{Float64}(undef, 0), n_frontier))
    append!(tree.r, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    return nothing
end
