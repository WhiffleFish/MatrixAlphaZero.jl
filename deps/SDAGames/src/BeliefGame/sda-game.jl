struct BeliefSDAGame{OBJ, UKF}
    rk4::RK4
    ts::FLOAT_RANGE
    epcs::Vector{Epoch}
    observer::Observer
    object::OBJ
    obs_noise::PDMat{Float64, SMatrix{4, 4, Float64, 16}}
    kf::UKF
    discount::Float64
    function BeliefSDAGame(; 
        ts=0.0:20.0:1200,    #6_000.0,
        epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0), 
        observer = Observer(
                sOSCtoCART(SA[R_EARTH+10_000e3, 0.5, 0.0, 0, 0, 100], use_degrees=true);
                ground_based = false
        ),
        object_of_interest = ObjectOfInterest(
            sOSCtoCART(SA[R_EARTH + 800e3, 0.0, 0.0, 0, 0, 0], use_degrees=true), 
            SA[1,1,1,1e-3,1e-3,1e-3],
            (
                (; ),
                (; drag=true, area_drag=1.0),
                (; drag=true, area_drag=5.0)
            )
        ),
        obs_noise = vcat(SA[100.0, 0.5], SA[1e-3, 1e-3]), # (10.0, 0.01), # (range, range_rate, azimuth, elevation)
        process_noise = vcat(@SArray(ones(3))*1e1, @SArray(ones(3))*1e0),
        discount = 0.95,
        n_grav = 0,
        m_grav = 0,
        drag = false,
        srp = false,
        moon = false,
        sun = false,
        relativity = false,
    )
        integrator_options = (;n_grav, m_grav, drag, srp, moon, sun, relativity)
        rk4 = RK4(fderiv_earth_orbit; integrator_options...)
        epcs = map(ts) do t
            epc0 + t
        end
        process_noise = convert_to_pdmat(process_noise)
        obs_noise = convert_to_pdmat(obs_noise)
        ukf = FilterBank(process_noise, obs_noise)
        return new{typeof(object_of_interest), typeof(ukf)}(rk4, ts, epcs, observer, object_of_interest, obs_noise, ukf, discount)
    end
end

struct SDAHistory{S,SO,B}
    t::Int
    sat_state::S
    observer_state::SO
    belief::B
end

function initialhist(game::BeliefSDAGame)
    return SDAHistory(1, mean(game.object.x), game.observer.x, game.object.x)
end

function gen_hist(game, s::SDAHistory, a)
    (; rk4) = game
    dt = step(game.ts)
    
    a_obs, a_sat = a
    epc = game.epcs[s.t]
    
    # propagate observer state
    observer_state = propagate_sat_state(game, epc, s.observer_state, a_obs)
    
    # propagate sat state
    sat_state = propagate_sat_state(game, epc, s.sat_state, a_sat)
    
    # propagate belief
    epcp = game.epcs[s.t+1]
    p = (;rk4, epc, dt, x_obs=observer_state)
    bpm = predict(game.kf, s.belief, a, p, s.t)
    o = gen_obs(game, observer_state, sat_state, epcp)
    bp = correct(game.kf, bpm, a, o, p, s.t)
    return SDAHistory(s.t+1, sat_state, observer_state, bp)
end

function belief_predict(game, epc, b, )
    dt = step(game.ts)
    rk4 = game.rk4
    predict(game.kf, b, nothing, (;rk4, epc, dt))
end

function apply_dv(x, Δv)
    Δvp = first(Δv)
    v = x[StaticArrays.SUnitRange(4,6)]
    v̂ = normalize(v, 2)
    return vcat(x[StaticArrays.SUnitRange(1,3)], v .+ Δvp .* v̂)
end

function propagate_sat_state(game, epc, x, a)
    x = apply_dv(x, a)
    dt = step(game.ts)
    return istep(game.rk4, epc, dt, x)
end

const NULL_RADAR_OBS = SA[NaN, NaN]
const NULL_CAMERA_OBS = SA[NaN, NaN]

isvalid_obs(y) = !all(isnan, y)

function gen_radar_obs(game, observer_state, sat_state, epc)
    return if can_radar_sat(epc, observer_state, sat_state)
        _gen_radar_obs(game, observer_state, sat_state)
    else
        NULL_RADAR_OBS
    end
end

function _gen_radar_obs(game, observer_state, sat_state)
    return single_obs(observer_state, sat_state) + rand(MvNormal(game.kf.radar.R2))
end

function gen_camera_obs(game, observer_state, sat_state, epc)
    return if can_see_sat(epc, observer_state, sat_state)
        _gen_camera_obs(game, observer_state, sat_state)
    else
        NULL_CAMERA_OBS
    end
end

function _gen_camera_obs(game, observer_state, sat_state)
    return azimuth_elevation(observer_state, sat_state) .+ rand(MvNormal(game.kf.cam.R2))
end

gen_obs(game, observer_state, sat_state, epc) = vcat(
    gen_radar_obs(game, observer_state, sat_state, epc), 
    gen_camera_obs(game, observer_state, sat_state, epc)
)
