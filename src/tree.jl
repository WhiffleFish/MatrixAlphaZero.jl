@proto struct Tree{S}
    s           ::  Vector{S}                           # [s_idx] -> s
    s_children  ::  Vector{Matrix{Int}}                 # [s_idx][a1, a2] -> sp_idx
    n_sa        ::  Vector{Matrix{Int}}                 # [s_idx][a1, a2] -> n_sa
    n_s         ::  Vector{Int}                         # [s_idx] -> n_s
    prior       ::  NTuple{2, Vector{Vector{Float32}}}  # [player][s_idx][a_i] -> π(s,a)
    v           ::  Vector{Matrix{Float64}}             # [s_idx][a1, a2] -> V(T(s, a1, a2))
    r           ::  Vector{Matrix{Float64}}             # [s_idx][a1, a2] -> r(s, a1, a2)
end

const NO_CHILDREN = Matrix{Int}(undef, 0, 0)
const NO_PRIOR = Vector{Float32}(undef, 0)

function Tree(game::MG, s=rand(initialstate(game)))
    return Tree(
        [s],                            # s
        Matrix{Int}[NO_CHILDREN],       # s_children
        Matrix{Int}[NO_CHILDREN],       # n_sa
        [0],                            # n_s
        ([NO_PRIOR], [NO_PRIOR]),       # prior
        [Matrix{Float64}(undef, 0, 0)], # v
        [Matrix{Float64}(undef, 0, 0)]  # r
    )
end

is_leaf(tree::Tree, s_idx::Int) = isempty(tree.s_children[s_idx])

function expand_s!(tree::Tree, s_idx::Int, game::MG, oracle)
    if is_leaf(tree, s_idx)
        _expand_s!(tree, s_idx, game, oracle)
    end
end

function _expand_s!(tree::Tree, s_idx::Int, game::MG, oracle)
    s = tree.s[s_idx]
    A1, A2 = actions(game)
    s_children = zeros(Int, length(A1), length(A2))
    r = zeros(Float64, length(A1), length(A2))
    v = zeros(Float64, length(A1), length(A2))
    counter = length(tree.s) + 1
    frontier = statetype(game)[]

    for (i,a1) ∈ enumerate(A1), (j,a2) ∈ enumerate(A2)
        s_children[i,j] = counter
        sp, r_i = @gen(:sp, :r)(game, s, (a1, a2))
        push!(frontier, sp)
        r[i,j] = r_i
        counter += 1
    end
    n_frontier = length(frontier)

    batch_sp = mapreduce(hcat, frontier) do s
        MarkovGames.convert_s(Vector{Float32}, s, game)
    end

    value = oracle(batch_sp)

    for i ∈ eachindex(value)
        v[i] = value[i]
    end
    
    tree.s_children[s_idx] = s_children
    tree.n_sa[s_idx] = zeros(Int, length(A1), length(A2))
    tree.n_s[s_idx] = 0
    # foreach(tree.prior, policy) do prior, π_i
    #     prior[s_idx] = π_i
    # end
    tree.v[s_idx] = v
    tree.r[s_idx] = r
    
    append!(tree.s, frontier)
    append!(tree.s_children, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_sa, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_s, fill(0, n_frontier))
    append!(tree.v, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.r, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    return nothing
end
