struct SearchTreeCore{S}
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
const NO_FLOAT = Vector{Float64}(undef, 0)
const CORE_TREE_FIELDS = (:s, :s_children, :n_sa, :n_s, :prior, :v, :r)

function SearchTreeCore(game::MG, s=rand(initialstate(game)))
    return SearchTreeCore(
        [s],
        Matrix{Int}[NO_CHILDREN],
        Matrix{Int}[NO_CHILDREN],
        [0],
        ([NO_PRIOR], [NO_PRIOR]),
        [Matrix{Float64}(undef, 0, 0)],
        [Matrix{Float64}(undef, 0, 0)],
    )
end

search_core(tree::AbstractSearchTree) = getfield(tree, :core)

function Base.getproperty(tree::AbstractSearchTree, sym::Symbol)
    if sym in CORE_TREE_FIELDS
        return getproperty(search_core(tree), sym)
    end
    return getfield(tree, sym)
end

function Base.propertynames(tree::AbstractSearchTree, private::Bool=false)
    fields = fieldnames(typeof(tree))
    return private ? (fields..., CORE_TREE_FIELDS...) : (filter(!=(:core), fields)..., CORE_TREE_FIELDS...)
end

is_leaf(tree::AbstractSearchTree, s_idx::Int) = isempty(tree.s_children[s_idx])

function Tree(params::MCTSParams, game::MG, s=rand(initialstate(game)))
    return Tree(params.search_style, game, s)
end

Tree(game::MG, s=rand(initialstate(game))) = Tree(RegretMatchingSearch(), game, s)

function reset_search_node! end
function append_search_frontier! end

function expand_s!(tree::AbstractSearchTree, s_idx::Int, game::MG, oracle)
    if is_leaf(tree, s_idx)
        _expand_s!(tree, s_idx, game, oracle)
    end
end

function _expand_s!(tree::AbstractSearchTree, s_idx::Int, game::MG, oracle)
    s = tree.s[s_idx]
    A1, A2 = actions(game)
    na1, na2 = length(A1), length(A2)
    s_children = zeros(Int, na1, na2)
    r = zeros(Float64, na1, na2)
    v = zeros(Float64, na1, na2)
    counter = length(tree.s) + 1
    frontier = statetype(game)[]
    nonterminal = trues(na1 * na2)

    flat_idx = 1
    for (j, a2) ∈ enumerate(A2), (i, a1) ∈ enumerate(A1)
        s_children[i, j] = counter
        sp, r_i = @gen(:sp, :r)(game, s, (a1, a2))
        push!(frontier, sp)
        r[i, j] = zs_reward_scalar(r_i)
        if isterminal(game, sp)
            nonterminal[flat_idx] = false
        end
        counter += 1
        flat_idx += 1
    end
    n_frontier = length(frontier)

    v̂ = batch_state_value(oracle, game, frontier)
    for i ∈ eachindex(v̂, nonterminal)
        v[i] = v̂[i] * nonterminal[i]
    end

    prior = state_policy(oracle, game, s)
    foreach(tree.prior, prior) do tree_prior, prior_i
        tree_prior[s_idx] = prior_i
    end

    tree.s_children[s_idx] = s_children
    tree.n_sa[s_idx] = zeros(Int, na1, na2)
    tree.n_s[s_idx] = 0
    tree.v[s_idx] = v
    tree.r[s_idx] = r
    reset_search_node!(tree, s_idx, na1, na2)

    append!(tree.s, frontier)
    append!(tree.s_children, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_sa, fill(NO_CHILDREN, n_frontier))
    append!(tree.n_s, fill(0, n_frontier))
    append!(tree.v, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    append!(tree.r, fill(Matrix{Float64}(undef, 0, 0), n_frontier))
    foreach(tree.prior) do prior_i
        append!(prior_i, fill(NO_PRIOR, n_frontier))
    end
    append_search_frontier!(tree, n_frontier)
    return nothing
end

function oracle_state_value(oracle, game::MG, s)
    return Float64(only(value(oracle, MarkovGames.convert_s(Vector{Float32}, s, game))))
end

function oracle_policy(params::MCTSParams, game::MG, tree::AbstractSearchTree, s_idx::Int)
    x, y = state_policy(params.oracle, game, tree.s[s_idx])
    return Float64.(x), Float64.(y)
end

function empirical_policy(tree::AbstractSearchTree, s_idx::Int)
    counts = tree.n_sa[s_idx]
    x = vec(sum(counts; dims=2))
    y = vec(sum(counts; dims=1))
    return normalize_or_uniform!(Float64.(x)), normalize_or_uniform!(Float64.(y))
end

function normalize_or_uniform!(x::AbstractVector)
    isempty(x) && return x
    s = sum(x)
    if s > 0
        x ./= s
    else
        x .= inv(length(x))
    end
    return x
end

eps_exploration(p, ϵ) = inv(length(p)) .* ϵ .+ (1 .- ϵ) .* p

function action_idx_from_probs(x, y)
    x = normalize_or_uniform!(Float64.(x))
    y = normalize_or_uniform!(Float64.(y))
    return CartesianIndex(
        rand(Categorical(x)),
        rand(Categorical(y)),
    )
end

function node_matrix_game(tree::AbstractSearchTree, s_idx::Int, γ::Float64)
    return tree.r[s_idx] .+ γ .* tree.v[s_idx]
end

node_return_sum(tree::AbstractBanditTree, s_idx::Int) = tree.return_sum[s_idx]

function add_return_sum!(tree::AbstractBanditTree, s_idx::Int, value::Float64)
    tree.return_sum[s_idx] += value
    return tree.return_sum[s_idx]
end
