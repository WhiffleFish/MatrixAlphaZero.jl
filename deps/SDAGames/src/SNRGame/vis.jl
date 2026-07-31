struct SNRGameFrames
    Xs::Vector{SVector{6, Float64}}
    Xo::Vector{SVector{6, Float64}}
    μs::Vector{Vector{Float64}}
    Σs::Vector{Matrix{Float64}}
    range::NTuple{2,Float64}
    rad_vis::BitVector
    cam_vis::BitVector
end

function SNRGameFrames(game::SNRSDAGame, s_hist)
    Xs = map(s_hist) do s
        s.sat_state
    end
    Xo = map(s_hist) do s
        s.observer_state
    end
    x_min = min(minimum(getindex.(Xs, 1)), minimum(getindex.(Xo, 1)))
    x_max = max(maximum(getindex.(Xs, 1)), maximum(getindex.(Xo, 1)))
    y_min = min(minimum(getindex.(Xs, 2)), minimum(getindex.(Xo, 2)))
    y_max = max(maximum(getindex.(Xo, 2)), maximum(getindex.(Xo, 2)))

    xrange = max(abs(x_min), abs(x_max))
    yrange = max(abs(y_min), abs(y_max))

    cam_vis = map(s_hist) do s
        can_see_sat(game.epcs[s.t], s.observer_state, s.sat_state)
    end

    return SNRGameFrames(
        Xs,
        Xo,
        μs,
        Σs,
        (xrange, yrange),
        rad_vis,
        cam_vis
    )
end

new_circle(x=0,y=0,r=1,n=100) = [(x + r * cos(u), y + r * sin(u)) for u in range(0, stop = 2π, length = n)]

@recipe function f(game::SNRSDAGame, s::SDAState; sqlim=nothing, sqsize=nothing, obj_marker_color=:red, tail_len=5, show_estimate=true, observability=true, showlabels=true)
    grid        --> false
    framestyle  --> :box
    fontfamily  --> "Computer Modern"
    labels      --> ""
    (; epc, observer, target) = s

    X_obs = s.observer
    X_rso = s.target
    cam = can_see_sat(epc, observer, target)
    
    if !isnothing(sqlim)
        xlims := (-sqlim, sqlim) .* 1.5
        ylims := (-sqlim, sqlim) .* 1.5
        aspect_ratio --> 1
    end
    
    observability && @series begin
        seriestype := :path
        linecolor := if cam
            :green
        else
            :black
        end
        # linestyle := :dash
        lw --> 3
        alpha --> 0.75
        [X_obs[1], X_rso[1]], [X_obs[2], X_rso[2]]
    end
    
    @series begin # Earth
        if !isnothing(sqlim)
            xlims := (-sqlim, sqlim) .* 1.5
            ylims := (-sqlim, sqlim) .* 1.5
            aspect_ratio --> 1
        end
        if !isnothing(sqsize)
            size --> ((2/3)*sqsize, sqsize)
            markersize --> sqsize / 100
            linewidth --> sqsize / 100
            legendfontsize --> sqsize / 50
        end
        seriestype := :shape
        fillcolor := :turquoise
        linewidth := 0
        new_circle(0, 0, R_EARTH, 100)
    end
    
    @series begin # scatter satellites
        seriestype := :path
        linecolor := nothing
        markercolor := [:blue :red] # why is this backwards???
        markershape --> :diamond
        labels  := ["Target" "Observer"]
        # labels  --> showlabels ? ["Observer" "Target"] : nothing
        [X_obs[1] X_rso[1]], [X_obs[2] X_rso[2]]
    end
    
end


@recipe function f(game::SNRSDAGame, hist::SimHistory)
    x_obs = Float64[]
    y_obs = Float64[]
    x_tar = Float64[]
    y_tar = Float64[]
    T = length(hist)

    for s_i ∈ hist[:s]
        push!(x_obs, s_i.observer[1])
        push!(y_obs, s_i.observer[2])
        push!(x_tar, s_i.target[1])
        push!(y_tar, s_i.target[2])
    end
    @series begin # Earth
        # if !isnothing(sqlim)
        #     xlims := (-sqlim, sqlim) .* 1.5
        #     ylims := (-sqlim, sqlim) .* 1.5
        #     aspect_ratio --> 1
        # end
        aspect_ratio --> 1
        # if !isnothing(sqsize)
        #     size --> ((2/3)*sqsize, sqsize)
        #     markersize --> sqsize / 100
        #     linewidth --> sqsize / 100
        #     legendfontsize --> sqsize / 50
        # end
        seriestype := :shape
        fillcolor := :turquoise
        linewidth := 0
        new_circle(0, 0, R_EARTH, 100)
    end

    @series begin
        alpha --> (1:T) ./ T
        lw --> 5
        c --> :blue
        x_obs, y_obs
    end
    @series begin
        alpha --> (1:T) ./ T
        lw --> 5
        c --> :red
        x_tar, y_tar
    end
end


# from HistoryRecorder
# show state + behavior
@recipe function f(game::SNRSDAGame, t::NamedTuple; showlabels=true)
    (;s, behavior) = t
    pol1 = behavior[1].probs |> permutedims
    pol2 = behavior[2].probs |> permutedims
    @series begin
        c       --> :blue
        lw      --> 10
        alpha   --> pol1
        labels  --> nothing
        action_lines(game, s, 1)
    end
    @series begin
        c       --> :red
        lw      --> 10
        alpha   --> pol2
        labels  --> nothing
        action_lines(game, s, 2)
    end
    @series begin
        showlabels --> showlabels
        game, s
    end
end


function action_lines(game::SNRSDAGame, x::SDAState, player::Int; n=5)
    (;epc) = x
    A = actions(game)[player]
    dts = range(0, game.dt, length=n)
    x = isone(player) ? x.observer : x.target
    return map(A) do a
        pts = map(dts) do dt
            sp = propagate_sat_state(game, epc, x, a; dt)
            sp[1], sp[2]
        end
        first.(pts), last.(pts)
    end
end
