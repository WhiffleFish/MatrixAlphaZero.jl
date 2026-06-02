struct SMOOSTree{S}
    s           :: Vector{S}
    children    :: Vector{Dict{Tuple{Int,Int}, Int}}
    regret      :: NTuple{2, Vector{Vector{Float64}}}
    strategy    :: NTuple{2, Vector{Vector{Float64}}}
end

function SMOOSTree(game::MG, s=rand(initialstate(game)))
    return SMOOSTree(
        [s],
        [Dict{Tuple{Int,Int}, Int}()],
        (Vector{Float64}[Float64[]], Vector{Float64}[Float64[]]),
        (Vector{Float64}[Float64[]], Vector{Float64}[Float64[]]),
    )
end

Tree(::SMOOSParams, game::MG, s=rand(initialstate(game))) = SMOOSTree(game, s)
Tree(game::MG, s=rand(initialstate(game))) = SMOOSTree(game, s)

function normalize_or_uniform!(x::AbstractVector)
    isempty(x) && return x
    s = sum(x)
    if isfinite(s) && s > 0
        x ./= s
    else
        x .= inv(length(x))
    end
    return x
end

normalized_or_uniform(x::AbstractVector) = normalize_or_uniform!(Float64.(copy(x)))

eps_exploration(p, ϵ) = inv(length(p)) .* ϵ .+ (1 .- ϵ) .* p

function match!(x, r)
    rs = zero(eltype(r))
    for i ∈ eachindex(x,r)
        r_i = r[i]
        if r_i > zero(r_i)
            rs += r_i
            x[i] = r_i
        else
            x[i] = zero(r_i)
        end
    end
    rs > zero(rs) ? x ./= rs : x .= inv(length(x))
end

function regret_matching_policy(regret::AbstractVector)
    policy = zeros(Float64, length(regret))
    return match!(policy, regret)
end

function action_idx_from_probs(x, y)
    x = normalized_or_uniform(x)
    y = normalized_or_uniform(y)
    return CartesianIndex(
        rand(Categorical(x)),
        rand(Categorical(y)),
    )
end

function oracle_state_value(oracle, game::MG, s)
    return Float64(only(value(oracle, MarkovGames.convert_s(Vector{Float32}, s, game))))
end

function expand_node!(tree::SMOOSTree, h::Int, game::MG, params::SMOOSParams)
    isempty(tree.regret[1][h]) || return nothing
    s = tree.s[h]
    A1, A2 = actions(game)
    r̂ = state_regret(params.oracle, game, s)
    ŝ = state_strategy(params.oracle, game, s)
    tree.regret[1][h] = params.τ .* Float64.(r̂[1])
    tree.regret[2][h] = params.τ .* Float64.(r̂[2])
    tree.strategy[1][h] = params.τ .* normalized_or_uniform(ŝ[1])
    tree.strategy[2][h] = params.τ .* normalized_or_uniform(ŝ[2])
    @assert length(tree.regret[1][h]) == length(A1)
    @assert length(tree.regret[2][h]) == length(A2)
    return nothing
end

function child_index!(tree::SMOOSTree, h::Int, a::CartesianIndex{2}, sp)
    key = Tuple(a)
    return get!(tree.children[h], key) do
        push!(tree.s, sp)
        push!(tree.children, Dict{Tuple{Int,Int}, Int}())
        push!(tree.regret[1], Float64[])
        push!(tree.regret[2], Float64[])
        push!(tree.strategy[1], Float64[])
        push!(tree.strategy[2], Float64[])
        return length(tree.s)
    end
end

function root_targets(params::SMOOSParams, tree::SMOOSTree, game::MG, h::Int=1)
    s = tree.s[h]
    expand_node!(tree, h, game, params)
    denom = params.τ + params.oos_iterations
    denom = denom > 0 ? denom : 1.0
    A1, A2 = actions(game)
    yr = (
        Float64.(tree.regret[1][h]) ./ denom,
        Float64.(tree.regret[2][h]) ./ denom,
    )
    ys = (
        Float64.(tree.strategy[1][h]) ./ denom,
        Float64.(tree.strategy[2][h]) ./ denom,
    )
    length(yr[1]) == length(A1) || error("player 1 regret target has wrong action dimension")
    length(yr[2]) == length(A2) || error("player 2 regret target has wrong action dimension")
    return yr, ys
end
