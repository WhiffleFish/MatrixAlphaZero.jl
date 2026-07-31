const Vec2{T} = SVector{2, T}
const Vec8 = SVector{8, Float64}

#player 1 is pursuer -- player 2 is evader

struct CTagState{T}
    pursuer::Vec2{T}
    evader::Vec2{T}
end

struct CTagBearingObs
    n::Int
    θ::Float64
    CTagBearingObs(n,θ=0.0) = new(n,θ)
end

function (obsmodel::CTagBearingObs)(θ)
    subdiv = 2π / obsmodel.n
    θrot = mod2pi(θ - obsmodel.θ)
    return Int(div(θrot, subdiv))
end

Base.@kwdef struct ContinuousTag{T,O<:Tuple} <: POMG{CTagState{T}, Tuple{Int,Int}, Tuple{Int, Int}}
    tag_radius::Float64                         = 0.1
    observations::O                             = (CTagBearingObs(4), CTagBearingObs(4))
    n_act::Int                                  = 4
    step_sizes::Tuple{Float64, Float64}         = (0.1, 0.1)
    initialstate_dims::Tuple{Float64, Float64}  = (0.25, 0.25)
    discount::Float64                           = 0.95
    dtype::Type{T}                              = Float32
    dense_reward::Bool                          = false
    transition_noise::Float64                   = 0.1
end

MarkovGames.discount(pomg::ContinuousTag) = pomg.discount
MarkovGames.actions(pomg::ContinuousTag) = (1:pomg.n_act, 1:pomg.n_act)
MarkovGames.observations(pomg::ContinuousTag) = map(obs -> 0:obs.n-1, pomg.observations)

function MarkovGames.initialstate(pomg::ContinuousTag{T}) where T
    dims = T.(pomg.initialstate_dims)
    return ImplicitDistribution() do rng
        s1 = (@SArray(rand(rng, T, 2)) .* 2 .- 1) .* dims
        s2 = (@SArray(rand(rng, T, 2)) .* 2 .- 1) .* dims
        CTagState{T}(s1, s2)
    end
end

function MarkovGames.observation(pomg::ContinuousTag, sp::CTagState)
    Δx = sp.evader - sp.pursuer
    θp = mod2pi(atan(Δx[2], Δx[1]))
    θe = mod2pi(θp + π)
    return ProductDistribution(
        Deterministic(pomg.observations[1](θp)), Deterministic(pomg.observations[2](θe))
    )
end

ctag_move(s::Vec2{T}, θ, ds) where T = SA[s[1] + T(ds*cos(θ)), s[2] + T(ds*sin(θ))]

function MarkovGames.gen(pomg::ContinuousTag, s::CTagState{T}, a::Tuple{Int, Int}, rng::Random.AbstractRNG=Random.default_rng()) where T
    subdiv = 2π / pomg.n_act
    θs = a .* subdiv
    θs = map(θs) do θ
        θ + pomg.transition_noise * randn(rng)
    end
    sp = CTagState{T}(
        ctag_move(s.pursuer, θs[1], pomg.step_sizes[1]),
        ctag_move(s.evader, θs[2], pomg.step_sizes[2]),
    )
    return (;sp)
end

function dense_pursuer_reward(pomg::ContinuousTag, s::CTagState, a)
    Δx = s.evader - s.pursuer
    return norm(Δx, 2)/pomg.tag_radius
end

function sparse_pursuer_reward(pomg::ContinuousTag, s::CTagState, a)
    Δx = s.evader - s.pursuer
    return norm(Δx, 2) ≤ pomg.tag_radius ? 1.0 : 0.0
end

function MarkovGames.reward(pomg::ContinuousTag, s::CTagState, a)
    r = pomg.dense_reward ? dense_pursuer_reward(pomg, s, a) : sparse_pursuer_reward(pomg, s, a)
    return SA[Float64(r), -Float64(r)]
end

## visualization -- old

function ctag_move_dirs(game, s)
    subdiv = 2π / game.n_act
    θs = subdiv:subdiv:2π
    xp, yp = s.pursuer
    xe, ye = s.evader
    Xsp = mapreduce(hcat, θs) do θ
        [xp, xp + game.step_sizes[1]*cos(θ)]
    end
    Ysp = mapreduce(hcat, θs) do θ
        [yp, yp + game.step_sizes[1]*sin(θ)]
    end
    Xse = mapreduce(hcat, θs) do θ
        [xe, xe + game.step_sizes[2]*cos(θ)]
    end
    Yse = mapreduce(hcat, θs) do θ
        [ye, ye + game.step_sizes[2]*sin(θ)]
    end
    Xs = hcat(Xsp, Xse)
    Ys = hcat(Ysp, Yse)
    return Xs, Ys
end

function tree_strategies(trees, node_idxs)
    return map(trees, node_idxs) do tree, node_idx
        node = tree.nodes[node_idx]
        normalize(node.s, 1)
    end
end

@recipe function ctag_vis(sol#=::ESCFRSolver=#, s::CTagState, node_idxs::Tuple{Int,Int}; labels=[""], strategies=true)
    game = sol.game
    xp, yp = s.pursuer
    xe, ye = s.evader
    Xs, Ys = ctag_move_dirs(game, s)
    σs = strategies ? tree_strategies(sol.trees, node_idxs) : nothing
    
    framestyle --> :box
    strategies && @series begin
        seriestype := :path
        c := permutedims(vcat(fill(:red,4), fill(:blue,4)))
        lw --> 20
        alpha := reduce(vcat, σs)'
        label := ""
        Xs, Ys
    end
    @series begin
        seriestype := :scatter
        c := [:red :blue]
        xlims           --> (-1,1)
        ylims           --> (-1,1)
        xticks          --> [-1,0,1]
        yticks          --> [-1,0,1]
        aspect_ratio    --> :equal
        ms              --> 10
        label           --> permutedims(vcat(fill("", 8), labels))
        [xp, xe]', [yp, ye]'
    end
end

## visualization -- monte carlo

next_policy_node(tree, node, a, o) = get(tree.children, (node, a, o), 0)

function action_from_node(tree, node, A, rng=Random.default_rng())
    if iszero(node)
        return rand(rng, A)
    else
        σ = normalize(tree.nodes[node].s, 1)
        a_idx = Solvers.weighted_sample_unnormed(rng,σ)
        return A[a_idx]
    end
end

function single_traj(game, sol, s; max_depth=sol.max_depth, rng=Random.default_rng(), nodes=(1,1))
    traj = [s]
    t = 0
    A = actions(game)
    while !isterminal(game, s) && t < max_depth
        a = map(sol.trees, nodes, A) do tree, node, Ai
            action_from_node(tree, node, Ai, rng)
        end
        sp, o = @gen(:sp, :o)(game, s, a)
        nodes = map(sol.trees, nodes, a, o) do tree, node, ai, oi
            next_policy_node(tree, node, ai, oi)
        end
        s = sp
        t += 1
        push!(traj, s)
    end
    return traj
end

function process_traj(traj#=::AbstractArray{<:POMGTreeExperiments.CTagState}=#)
    Xp = map(traj) do s
        s.pursuer[1]
    end
    Yp = map(traj) do s
        s.pursuer[2]
    end
    Xe = map(traj) do s
        s.evader[1]
    end
    Ye = map(traj) do s
        s.evader[2]
    end
    return (Xp, Yp), (Xe, Ye)
end

function multi_traj_data(game, sol, s, N; kwargs...)
    Xps = []
    Yps = []
    Xes = []
    Yes = []
    for _ ∈ 1:N
        (Xp, Yp), (Xe, Ye) = process_traj(single_traj(game, sol, s; kwargs...))
        push!(Xps, Xp)
        push!(Yps, Yp)
        push!(Xes, Xe)
        push!(Yes, Ye)
    end
    return reduce(hcat,Xps), reduce(hcat,Yps), reduce(hcat,Xes), reduce(hcat,Yes)
end

struct CTagMCData
    s::CTagState
    Sp::NTuple{2, AbstractMatrix}
    Se::NTuple{2, AbstractMatrix}
    function CTagMCData(game::ContinuousTag, sol#=::ESCFRSolver=#, s::CTagState; N=1_000, kwargs...)
        Xp, Yp, Xe, Ye = multi_traj_data(game, sol, s, N; kwargs...)
        return new(s, (Xp, Yp), (Xe, Ye))
    end
end

function ctag_mc_sim(game::ContinuousTag, sol, s; nodes=(1,1), rng=Random.default_rng(), N=1000)
    A = actions(game)
    data_buffer = []
    for i ∈ 0:(sol.max_depth-1)
        mc_data = CTagMCData(game, sol, s; nodes, max_depth=sol.max_depth-i, N=N)
        a = map(sol.trees, nodes, A) do tree, node, Ai
            action_from_node(tree, node, Ai, rng)
        end
        sp, o = @gen(:sp, :o)(game, s, a)
        nodes = map(sol.trees, nodes, a, o) do tree, node, ai, oi
            next_policy_node(tree, node, ai, oi)
        end
        s = sp
        push!(data_buffer, mc_data)
    end
    return data_buffer
end

@recipe function ctag_mc_vis(data::CTagMCData; strategies=true, alpha=0.1)
    s = data.s
    xp, yp = s.pursuer
    xe, ye = s.evader
    Xp, Yp = data.Sp
    Xe, Ye = data.Se
    
    framestyle --> :box
    labels --> ""
    if strategies
        @series begin
            seriestype := :path
            c := 2
            lw --> 1
            alpha := alpha
            label := ""
            Xp, Yp
        end
        @series begin
            seriestype := :path
            c := 1
            lw --> 1
            alpha := alpha
            label := ""
            Xe, Ye
        end
    end
    @series begin
        seriestype := :scatter
        c := [:red :blue]
        xlims           --> (-1,1)
        ylims           --> (-1,1)
        xticks          --> [-1,0,1]
        yticks          --> [-1,0,1]
        aspect_ratio    --> :equal
        ms              --> 10
        # label           --> permutedims(vcat(fill("", 8), labels))
        [xp, xe]', [yp, ye]'
    end
end

# @recipe function f(b::UnweightedParticleCollection{<:CTagState})
#     b_p = vecvec2mat(getfield.(b.particles, :pursuer))
#     b_e = vecvec2mat(getfield.(b.particles, :evader))
#     @series begin
#         seriestype := :scatter
#         b_p[:,1], b_p[:,2]
#     end
#     @series begin
#         seriestype := :scatter
#         b_e[:,1], b_e[:,2]
#     end
#     @series begin
#         alpha   --> 0.1
#         c       --> 1 
#         ls      --> :dash
#         hcat(b_p[:,1], b_e[:,1])', hcat(b_p[:,2], b_e[:,2])'
#     end
# end
