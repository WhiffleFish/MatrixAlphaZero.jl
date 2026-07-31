struct SDAObsDist{NO, NTOT}
    t::NTuple{NO, SDAStaticSingleObsDist}
    function SDAObsDist{NTOT}(t::NTuple{NO, SDAStaticSingleObsDist}) where {NO,NTOT}
        new{NO,NTOT}(t)
    end
end

Random.gentype(::Type{SDAObsDist{NO, NTOT}}) where {NO, NTOT} = Observation{NO,NTOT}

function POMDPs.pdf(dist::SDAObsDist, o::AbstractMatrix)
    return mapreduce(*, dist.t, eachcol(o)) do d_i, o_i
        pdf(d_i, o_i)
    end
end

function Distributions.logpdf(dist::SDAObsDist, o::AbstractMatrix)
    return mapreduce(+, dist.t, eachcol(o)) do d_i, o_i
        logpdf(d_i, o_i)
    end
end

function static_rand(rng::AbstractRNG, d::SDAStaticSingleObsDist) # MvNormal
    μ = d.μ
    Σ = diag(d.Σ)
    return μ .+ @SArray(rand(rng, 2)) .* Σ
end

function Base.rand(rng::AbstractRNG, s::Random.SamplerTrivial{<:SDAObsDist})
    return mapreduce(hcat,s[].t) do d_i
        static_rand(rng, d_i)
    end
end

function POMDPs.observation(pomdp::SDAPOMDP, a, sp)
    (; X, t) = sp
    X_obs = map(pomdp.observers) do x
        @view x[:,t]
    end
    return map(X_obs, Tuple(a)) do x_obs, rso_idx
        single_obs_dist(pomdp, x_obs, X[:, rso_idx])
    end |> SDAObsDist{obsmatsize(pomdp)}
end

function full_obs(pomdp::SDAPOMDP, a, sp)
    (; X, t) = sp
    X_obs = map(pomdp.observers) do x
        @view x[:,t]
    end
    return mapreduce(hcat, X_obs, Tuple(a)) do x_obs, rso_idx
        single_obs(x_obs, X[:, rso_idx])
    end
end

function single_obs(x_obs, x_rso)
    Rs = x_obs[1:3]
    Vs = x_obs[4:6]
    R = x_rso[1:3]
    V = x_rso[4:6]
    return SA[
        station_range(R, Rs, V, Vs),
        station_range_rate(R, Rs, V, Vs)
    ]
end

function single_obs_dist(pomdp::SDAPOMDP, x_obs, x_rso)
    return MvNormal(single_obs(x_obs, x_rso), pomdp.obs_noise)
end

function station_view_angle(R, Rs)
    R_rel = R - Rs
    return angle_between(R_rel, Rs)
end

masked(R, Rs) = station_view_angle(R, Rs) > deg2rad(80)

station_range(R, Rs, V, Vs) = norm(R - Rs, 2)
station_range(X::AbstractVector, Xs::AbstractVector) = station_range(X[1:3], Xs[1:3], X[4:6], Xs[4:6])

function station_range_rate(R, Rs, V, Vs)
    return dot(R - Rs, V - Vs) / station_range(R, Rs, V, Vs)
end

station_range_rate(X::AbstractVector, Xs::AbstractVector) = station_range_rate(X[1:3], Xs[1:3], X[4:6], Xs[4:6])

function station_range_∇R(R, Rs, V, Vs)
    return (R - Rs) ./ station_range(R, Rs, V, Vs)
end

station_range_∇V(R, Rs, V, Vs) = zero(V)

function station_range_rate_∇R(R, Rs, V, Vs)
    ρ = station_range(R, Rs, V, Vs)
    return (ρ * (V - Vs) - dot(R - Rs, V - Vs) * (R - Rs) /  ρ ) / ρ^2
end

function station_range_rate_∇V(R, Rs, V, Vs)
    ρ = station_range(R, Rs, V, Vs)
    return (R - Rs) / ρ
end

function station_range_∇Rs(R, Rs, V, Vs)
    return (Rs - R) ./ station_range(R, Rs, V, Vs)
end

station_range_∇Vs(R, Rs, V, Vs) = zero(Vs)

function station_range_rate_∇Rs(R, Rs, V, Vs)
    ρ = station_range(R, Rs, V, Vs)
    return (ρ * (Vs - V) - dot(R - Rs, V - Vs) * (Rs - R) /  ρ ) / ρ^2
end

function station_range_rate_∇Vs(R, Rs, V, Vs)
    ρ = station_range(R, Rs, V, Vs)
    return (Rs - R) / ρ
end

function azimuth_elevation(R::AbstractVector, Rs::AbstractVector)
    r̂ = normalize(Rs - R, 2)
    az = atan(r̂[2], r̂[1])
    el = asin(r̂[3])
    return az, el
end

##

radar_measurement(x_sat, u, p, t=0) = _radar_measurement(p.x_obs, x_sat)

function _radar_measurement(x_obs, x_sat)
    Rs = x_obs[StaticArrays.SUnitRange(1,3)]
    Vs = x_obs[StaticArrays.SUnitRange(4,6)]
    R = x_sat[StaticArrays.SUnitRange(1,3)]
    V = x_sat[StaticArrays.SUnitRange(4,6)]
    return SA[
        station_range(R, Rs, V, Vs),
        station_range_rate(R, Rs, V, Vs)
    ]
end

camera_measurement(x_sat, u, p, t=0) = _camera_measurement(p.x_obs, x_sat)

_camera_measurement(x_obs::AbstractVector, x_sat::AbstractVector) = SA[azimuth_elevation(x_obs, x_sat)...]

joint_measurement(x_sat, u, p, t=0) = _joint_measurement(p.x_obs, x_sat)

_joint_measurement(x_obs, x_sat) = vcat(
    _radar_measurement(x_obs, x_sat), 
    _camera_measurement(x_obs, x_sat)
)
