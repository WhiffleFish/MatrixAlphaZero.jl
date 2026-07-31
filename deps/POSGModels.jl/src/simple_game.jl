module SimpleGame

using MarkovGames
using Random
using POMDPTools
using StaticArrays

export SimpleMG

const SimpleState = Vector{Tuple{Int, Int}}

push(v::Array, x)= push!(copy(v), x)

next_state(s, a1, a2) = push(s, (a1, a2))

function traverse!(rewards::Dict, s::Vector, d::Int, na::Tuple, max_depth::Int)
    na1, na2 = na
    rewards[s] = randn(na...)
    if d ≥ max_depth
        return nothing
    else
        for a1 ∈ 1:na1, a2 ∈ 1:na2
            sp = next_state(s, a1, a2)
            traverse!(rewards, sp, d+1, na, max_depth)
        end
    end
end

#=
D = 2
na1 = 2, na2 = 2
(0,0) => [0,0, 0,0, 0,0]
(1,1) => [1,1, 0,0, 0,0]
[(1,1), (2,1)] => [1,1, 2,1, 0,0]
=#
struct SimpleMG <: MG{SimpleState, Tuple{Int, Int}}
    rewards::Dict{Vector{Tuple{Int, Int}}, Matrix{Float64}}
    max_depth::Int
    discount::Float64
    function SimpleMG(na1, na2; discount=0.95, max_depth=2)
        # depth 0
        rewards = Dict{SimpleState, Matrix{Float64}}()
        traverse!(rewards, Tuple{Int,Int}[], 0, (na1, na2), max_depth)
        new(rewards, max_depth, discount)
    end
end

MarkovGames.initialstate(::SimpleMG) = Deterministic(Tuple{Int,Int}[])

function MarkovGames.reward(p::SimpleMG, s, a::Tuple)
    r = isterminal(p, s) ? 0.0 : p.rewards[s][a...]
    return SA[Float64(r), -Float64(r)]
end

MarkovGames.actions(p::SimpleMG) = axes(first(values(p.rewards)))

MarkovGames.states(p::SimpleMG) = push!(collect(keys(p.rewards)), [(-1,-1)])

MarkovGames.discount(p::SimpleMG) = p.discount

function MarkovGames.transition(p::SimpleMG, s, a)
    return if isterminal(p, s) # already terminal
        [(-1,-1)]
    elseif length(s) == p.max_depth # to terminal
        [(-1,-1)]
    else
        next_state(s, a...)
    end |> Deterministic
end

MarkovGames.isterminal(::SimpleMG, s) = !isempty(s) && first(first(s)) === -1

function MarkovGames.convert_s(::Type{Vector{T}}, s::SimpleState, p::SimpleMG) where T
    na1, na2 = length.(actions(p))
    l = p.max_depth*(na1+na2)
    if isterminal(p, s)
        return ones(T, l) .* -1
    else
        A = actions(p)
        na1, na2 = length.(A)
        v = zeros(T, l)
        step = na1 + na2
        for i ∈ eachindex(s)
            a = s[i]
            p1_idx = step * (i-1)
            a1_idx = p1_idx + a[1]
            v[a1_idx] = one(T)
            
            p2_idx = p1_idx + na1
            a2_idx = p2_idx + a[2]
            v[a2_idx] = one(T)
        end
        return v
    end

    # return mapreduce(vcat, s) do a 
    #     v1 = zeros(T, na1)
    #     v1[a[1]] = one(T)
    #     v2 = zeros(T, na2)
    #     v2[a[2]] = one(T)
    #     vcat(v1,v2) # vcat inside vcat inside vcat inside vcat inside vcat inside vcat
    # end
end

# function MarkovGames.convert_s(::Type{Int}, v::AbstractVector, p::SimpleMG)
#     return findfirst(isone, v) - 1
# end

struct SolutionNode
    policy::NTuple{2, Vector{Float64}}
    r::Matrix{Float64}
    vp::Matrix{Float64}
    Q::Matrix{Float64}
    value::Float64
end

function exact_solve(matrix_solver, game::SimpleMG)
    solution = Dict{SimpleState, SolutionNode}()
    s = rand(initialstate(game))
    solve_traverse!(solution, matrix_solver, game, s)
    return solution
end

function solve_traverse!(solution::Dict, solver, game, s::Vector)
    γ = discount(game)
    if isterminal(game, s)
        return 0.0
    else
        A1, A2 = actions(game)
        R = zeros(length(A1), length(A2))
        Q = zeros(length(A1), length(A2))
        VP =zeros(length(A1), length(A2))
        for i ∈ eachindex(A1), j ∈ eachindex(A2)
            a1, a2 = A1[i], A2[j]
            sp, r = @gen(:sp, :r)(game, s, (a1, a2))
            vp = solve_traverse!(solution, solver, game, sp)
            r1 = r[1]
            R[i,j] = r1
            VP[i,j] = vp
            Q[i,j] = r1 + γ*vp
        end
        x,y,t = solve(solver, Q)
        solution[s] = SolutionNode((x,y), R, VP, Q, t)
        return t
    end
end

## Exact Exploitability

struct ExploitNode
    policy::NTuple{2, Vector{Float64}}
    exploit_policy::NTuple{2, Int}
    r::Matrix{Float64}
    vp::NTuple{2,Matrix{Float64}}
    Q::NTuple{2,Matrix{Float64}}
    exploit_value::NTuple{2,Float64}
    attained_value::NTuple{2,Float64}
end

function exact_exploitability(planner, game::SimpleMG)
    solution = Dict{SimpleState, ExploitNode}()
    s = rand(initialstate(game))
    solve_traverse!(solution, planner, game, s)
    return solution
end

function exploit_traverse!(solution::Dict, planner, game, s::Vector)
    γ = discount(game)
    if isterminal(game, s)
        return 0.0, 0.0
    else
        A1, A2 = actions(game)
        R = zeros(length(A1), length(A2))
        Q1 = zeros(length(A1), length(A2))
        Q2 = zeros(length(A1), length(A2))
        VP1 =zeros(length(A1), length(A2))
        VP2 =zeros(length(A1), length(A2))
        for i ∈ eachindex(A1), j ∈ eachindex(A2)
            a1, a2 = A1[i], A2[j]
            sp, r = @gen(:sp, :r)(game, s, (a1, a2))
            vp1, vp2 = exploit_traverse!(solution, planner, game, sp)
            r1, r2 = r
            R[i,j] = r1
            VP1[i,j] = vp1
            VP2[i,j] = vp2
            Q1[i,j] = r1 + γ*vp1
            Q2[i,j] = r2 + γ*vp2
        end
        b = behavior(planner, s) # assuming sparsecat
        @assert b[1] isa SparseCat
        @assert b[2] isa SparseCat
        x,y = b[1].probs, b[2].probs
        
        Qx = Q1 * y
        v1 = maximum(Qx)
        Qy = Q2' * x
        v2 = maximum(Qy)

        solution[s] = ExploitNode((x,y), R, VP, Q, t)
        return v1, v2
    end
end

function exact_brv(planner, game::SimpleMG, player=1)
    s = rand(initialstate(game))
    return brv_traverse!(planner, game, s, player)
end

# what is the maximum utility `player` can get?
function brv_traverse!(planner, game, s::Vector, player)
    γ = discount(game)
    if isterminal(game, s)
        return 0.0
    else
        A1, A2 = actions(game)
        Q = zeros(length(A1), length(A2))
        for i ∈ eachindex(A1), j ∈ eachindex(A2)
            a1, a2 = A1[i], A2[j]
            sp, r = @gen(:sp, :r)(game, s, (a1, a2))
            vp = brv_traverse!(planner, game, sp, player)
            Q[i,j] = r[player] + γ*vp
        end
        b = behavior(planner, s) # assuming sparsecat
        @assert b[1] isa SparseCat
        @assert b[2] isa SparseCat
        x,y = b[1].probs, b[2].probs
        return if isone(player)
            maximum(Q * y)
        else
            maximum(Q' * x)
        end
    end
end



end
