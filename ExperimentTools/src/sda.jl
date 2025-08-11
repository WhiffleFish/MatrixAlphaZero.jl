struct OracleInfo
    obs_x::Vector{Float64}
    obs_y::Vector{Float64}
    s_target::Vector{Vector{Float64}}
    vals::Vector{Vector{Float64}}
    valrange::NTuple{2, Float64}
    sun_pos::NTuple{2, Vector{Float64}}
end

Base.eachindex(info::OracleInfo) = eachindex(info.s_target)

function OracleInfo(game, oracle, d_observer, θs = LinRange(0, 2π, 100); r_target = 5e6, n=1000)
    s_observers = rand(d_observer, n)
    oracle_info = map(θs) do θ
        s_target = sOSCtoCART([
            R_EARTH + r_target,
            0.0,
            0.0,
            0.0,
            0.0,
            θ
        ])
        s_states = map(s_observers) do s_o
            SNRGame.SDAState(s_o, s_target, game.epc0, false)
        end
        states_batch = mapreduce(hcat, s_states) do s
            MarkovGames.convert_s(Vector{Float32}, s, game)
        end
        oracle_values = Float64.(AZ.value(oracle, states_batch))
        return (;s_target, oracle_values)
    end
    vals = getfield.(oracle_info, :oracle_values)
    s_target = getfield.(oracle_info, :s_target)
    min_val, max_val = minimum(minimum.(vals)), maximum(maximum.(vals))
    # xl = xlims(p)
    # yl = ylims(p)
    # xrange = xl[2] - xl[1]
    # yrange = yl[2] - yl[1]
    sun_vec = sun_position(game.epc0)
    v̂s = normalize(sun_vec[1:2], 2)
    v0 = [0,0]
    vf = v0 .+ v̂s # .* xrange .* 0.1
    obs_x = getindex.(s_observers, 1)
    obs_y = getindex.(s_observers, 2)
    return OracleInfo(
        obs_x, obs_y, s_target, vals, (min_val, max_val), (v0, vf)
    )
end

struct OracleInfoFrame
    obs_x::Vector{Float64}
    obs_y::Vector{Float64}
    s_target::Vector{Float64}
    vals::Vector{Float64}
    valrange::NTuple{2, Float64}
    sun_pos::NTuple{2, Vector{Float64}}
end

Base.getindex(info::OracleInfo, i::Int) = OracleInfoFrame(
    info.obs_x,
    info.obs_y,
    info.s_target[i],
    info.vals[i],
    info.valrange,
    info.sun_pos
)

@recipe function f(info::OracleInfoFrame)
    @series begin
        xticks --> [-1.5e7, 0, 1.5e7]
        yticks --> [-1.5e7, 0, 1.5e7]
        clims --> info.valrange,
        ms --> 5
        alpha --> 0.5
        aspect_ratio --> 1.0
        size --> (800, 800)
        
        seriestype := :scatter
        markerstrokewidth := 0

        zcolor := info.vals
        info.obs_x, info.obs_y
    end
    @series begin
        seriestype := :scatter
        ms --> 10
        [info.s_target[1]], [info.s_target[2]]
    end
    # @series begin
    #     arrow := true
    #     lw --> 5
    #     [sun_pos[1][1], sun_pos[2][1]], [sun_pos[1][2], sun_pos[2][2]]
    # end
end
