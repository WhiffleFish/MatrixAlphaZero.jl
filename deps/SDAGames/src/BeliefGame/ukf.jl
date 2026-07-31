function set_state!(ukf::UnscentedKalmanFilter, b)
    ukf.x = mean(b)
    ukf.R = cov(b)
end

function get_state(ukf::UnscentedKalmanFilter)
    return MvNormal(LowLevelParticleFilters.state(ukf), covariance(ukf))
end

function predict(kf::UnscentedKalmanFilter, b::MvNormal, u, p, t::Real=0)
    set_state!(kf, b)
    LLPF.predict!(kf, u, p, t)
    return get_state(kf)
end

function correct(kf::UnscentedKalmanFilter, u, y, p, t)
    LLPF.correct!(kf, u, y, p, t)
    return get_state(kf)
end

function update(kf::UnscentedKalmanFilter, b::MvNormal, a, o, p, t=0)
    set_state!(kf, b)
    LLPF.predict!(kf, u, p, t)
    LLPF.correct!(kf, u, y, p, t)
    return get_state(ukf)
end


##

radar_ukf(process_noise::PDMat, obs_noise::PDMat) = UnscentedKalmanFilter(
    dynamics, radar_measurement, process_noise, obs_noise, ny=2, nu=1
)

camera_ukf(process_noise::PDMat, obs_noise::PDMat) = UnscentedKalmanFilter(
    dynamics,
    UKFMeasurementModel{Float64, false, false}(
        camera_measurement, obs_noise; 
        nx=6, ny=2,
        innovation=cam_innovation, 
        mean=camera_mean_meas
    ),
    process_noise, ny=2, nu=1
)

joint_ukf(process_noise::PDMat, obs_noise::PDMat) = UnscentedKalmanFilter(
    dynamics, 
    UKFMeasurementModel{Float64, false, false}(
        joint_measurement, obs_noise; 
        nx=6, ny=4,
        innovation = joint_innovation, 
        mean = joint_mean_meas
    ),
    process_noise, ny=4, nu=1
)

struct FilterBank{R<:UnscentedKalmanFilter,C<:UnscentedKalmanFilter,B<:UnscentedKalmanFilter}
    radar::R
    cam::C
    both::B
end

function FilterBank(process_noise::PDMat, obs_noise::PDMat)
    return FilterBank(
        radar_ukf(process_noise, PDMat(obs_noise[StaticArrays.SUnitRange(1,2), StaticArrays.SUnitRange(1,2)])),
        camera_ukf(process_noise, PDMat(obs_noise[StaticArrays.SUnitRange(3,4), StaticArrays.SUnitRange(3,4)])),
        joint_ukf(process_noise, obs_noise)
    )
end

predict(bank::FilterBank, args...) = predict(bank.both, args...)

function correct(bank::FilterBank, b, u, y::AbstractVector, p, t)
    y_r = y[StaticArrays.SUnitRange(1,2)]
    y_c = y[StaticArrays.SUnitRange(3,4)]
    rad = isvalid_obs(y_r)
    cam = isvalid_obs(y_c)
    if rad && cam
        set_state!(bank.both, b)
        return correct(bank.both, u, y, p, t)
    elseif rad
        set_state!(bank.radar, b)
        return correct(bank.radar, u, y_r, p, t)
    elseif cam
        set_state!(bank.cam, b)
        return correct(bank.cam, u, y_c, p, t)
    else
        return b
    end
end

##
# https://c4i.gmu.edu/~pcosta/F15/data/fileserver/file/472121/filename/Paper_1570110549.pdf

wrappi(x) = mod2pi(x + π) - π

angle_innovation(θ, θm) = wrappi(θ) - wrappi(θm)

cam_innovation(x, xm) = SA[
    angle_innovation(x[1], xm[1])   # azimuth
    x[2] - xm[2]                    # elevation
]

joint_innovation(x, xm) = SA[
    x[1] - xm[1],                   # range 
    x[2] - xm[2],                   # range rate
    angle_innovation(x[1], xm[1]),  # azimuth
    x[4] - xm[4]                    # elevation
]

circular_mean(θs) = atan(sum(sin, θs), sum(cos, θs))

camera_mean_meas(x) = SA[
    circular_mean(getindex.(x, 1))  # azimuth 
    LLPF.safe_mean(getindex.(x, 2)) # elevation
]

joint_mean_meas(x) = SA[
    LLPF.safe_mean(getindex.(x, 1)) # range
    LLPF.safe_mean(getindex.(x, 2)) # range rate
    circular_mean(getindex.(x, 3))  # azimuth 
    LLPF.safe_mean(getindex.(x, 4)) # elevation
]
