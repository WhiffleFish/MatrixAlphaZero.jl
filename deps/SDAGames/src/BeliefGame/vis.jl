struct SDAGameFrames
    Xs::Vector{SVector{6, Float64}}
    Xo::Vector{SVector{6, Float64}}
    μs::Vector{Vector{Float64}}
    Σs::Vector{Matrix{Float64}}
    range::NTuple{2,Float64}
    rad_vis::BitVector
    cam_vis::BitVector
    function SDAGameFrames(game, s_hist)
        Xs = map(s_hist) do s
            s.sat_state
        end
        Xo = map(s_hist) do s
            s.observer_state
        end
        μs = map(s_hist) do s
            mean(s.belief)
        end
        Σs = map(s_hist) do s
            cov(s.belief)
        end
        x_min = min(minimum(getindex.(Xs, 1)), minimum(getindex.(Xo, 1)))
        x_max = max(maximum(getindex.(Xs, 1)), maximum(getindex.(Xo, 1)))
        y_min = min(minimum(getindex.(Xs, 2)), minimum(getindex.(Xo, 2)))
        y_max = max(maximum(getindex.(Xo, 2)), maximum(getindex.(Xo, 2)))

        xrange = max(abs(x_min), abs(x_max))
        yrange = max(abs(y_min), abs(y_max))

        rad_vis = map(s_hist) do s
            can_radar_sat(game.epcs[s.t], s.observer_state, s.sat_state)
        end

        cam_vis = map(s_hist) do s
            can_see_sat(game.epcs[s.t], s.observer_state, s.sat_state)
        end

        return new(
            Xs,
            Xo,
            μs,
            Σs,
            (xrange, yrange),
            rad_vis,
            cam_vis
        )
    end
end

new_circle(x=0,y=0,r=1,n=100) = [(x + r * cos(u), y + r * sin(u)) for u in range(0, stop = 2π, length = n)]

function covellipse_points(μ, Σ)
    μ, S = StatsPlots._covellipse_args((μ, Σ); n_std=1)
    θ = range(0, 2π; length = 100)
    A = S * [cos.(θ)'; sin.(θ)']
    return μ[1] .+ A[1, :], μ[2] .+ A[2, :]
end

@recipe function f(frames::SDAGameFrames, t::Int; sqlim=nothing, sqsize=nothing, obj_marker_color=:red, tail_len=5, show_estimate=true, observability=true)
    grid        --> false
    framestyle  --> :box
    fontfamily  --> "Computer Modern"
    labels      --> ""
    layout := @layout [
        traj{0.66h}
        pos_est vel_est
    ]
    est_err = frames.μs[t] - frames.Xs[t]
    (; Xs, Xo) = frames
    sqlim = maximum(frames.range)
    sat_traj_tail = Xs[max(t - tail_len, 1):t]
    obs_traj_tail = Xo[max(t - tail_len, 1):t]

    X_obs = frames.Xo[t]
    X_rso = frames.Xs[t]

    rad = frames.rad_vis[t]
    cam = frames.cam_vis[t]
    
    @series begin # Earth
        subplot := 1
        if !isnothing(sqlim)
            subplot := 1
            xlims := (-sqlim, sqlim) .* 1.5
            ylims := (-sqlim, sqlim) .* 1.5
            aspect_ratio --> 1
        end
        if !isnothing(sqsize)
            subplot := 1
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
        subplot := 1
        seriestype := :path
        linecolor := nothing
        markercolor := [:red :blue]
        markershape --> :diamond
        [X_obs[1] X_rso[1]], [X_obs[2] X_rso[2]]
    end
    observability && (cam || rad) && @series begin
        subplot := 1
        seriestype := :path
        linecolor := if cam && rad
            :purple
        elseif cam
            :blue
        elseif rad
            :red
        else
            :black
        end
        linestyle := :dash
        [X_obs[1], X_rso[1]], [X_obs[2], X_rso[2]]
    end
    @series begin # path tails
        subplot := 1
        seriestype := :path
        xlims := (-sqlim, sqlim) .* 1.1
        ylims := (-sqlim, sqlim) .* 1.1
        
        # but like fuckin why tho??? - what about including observability lines causes color switch??
        linecolor := (observability && (rad || cam)) ? [:blue :red] : [:red :blue]
        if observability 
            annotation := [
                (0.90sqlim, 0.95sqlim, text("RAD", 10, rad ? :green : :red))
                (0.90sqlim, 0.80sqlim, text("CAM", 10, cam ? :green : :red))
            ]
        end
        (
            [getindex.(obs_traj_tail, 1) getindex.(sat_traj_tail, 1)], 
            [getindex.(obs_traj_tail, 2) getindex.(sat_traj_tail, 2)]
        )
    end
    ec_pts_pos = covellipse_points([0,0], frames.Σs[t][1:2, 1:2])
    ec_pts_vel = covellipse_points([0,0], frames.Σs[t][4:5, 4:5])
    
    x_min_e, x_max_e = extrema(first(ec_pts_pos))
    x_min_e, x_max_e = min(x_min_e, est_err[1]), max(x_max_e, est_err[1])
    y_min_e, y_max_e = extrema(last(ec_pts_pos))
    y_min_e, y_max_e = min(y_min_e, est_err[2]), max(y_max_e, est_err[2])

    vx_min_e, vx_max_e = extrema(first(ec_pts_vel))
    vx_min_e, vx_max_e = min(vx_min_e, est_err[4]), max(vx_max_e, est_err[4])
    vy_min_e, vy_max_e = extrema(last(ec_pts_vel))
    vy_min_e, vy_max_e = min(vy_min_e, est_err[5]), max(vy_max_e, est_err[5])
    tick_sigdigits = 2

    @series begin # x,y error covariance
        aspect_ratio := 1
        subplot := 2
        seriestype := :shape
        seriesalpha --> 0.3
        ec_pts_pos
    end
    @series begin # x,y error
        subplot := 2
        seriestype := :scatter
        # xlabel := L"e_{x}"
        # ylabel := L"e_{y}"
        title := L"e_x"
        xformatter := :scientific
        yformatter := :scientific
        xticks := [round(x_min_e, sigdigits=tick_sigdigits), 0, round(x_max_e, sigdigits=tick_sigdigits)]
        yticks := [round(y_min_e, sigdigits=tick_sigdigits), 0, round(y_max_e, sigdigits=tick_sigdigits)]
        [est_err[1]], [est_err[2]]
    end
    @series begin # vx,vy error covariance
        aspect_ratio := 1
        subplot := 3
        seriestype := :shape
        seriesalpha --> 0.3
        ec_pts_vel
    end
    @series begin # vx,vy error
        subplot := 3
        seriestype := :scatter
        # xlabel := L"e_{v_x}"
        # ylabel := L"e_{v_y}"
        title := L"e_{\dot{x}}"
        xformatter := :scientific
        yformatter := :scientific
        xticks := [round(vx_min_e, sigdigits=tick_sigdigits), 0, round(vx_max_e, sigdigits=tick_sigdigits)]
        yticks := [round(vy_min_e, sigdigits=tick_sigdigits), 0, round(vy_max_e, sigdigits=tick_sigdigits)]
        [est_err[4]], [est_err[5]]
    end
end
