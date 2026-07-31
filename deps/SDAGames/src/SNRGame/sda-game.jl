struct SDAState
    observer    :: StateVec
    target      :: StateVec
    epc         :: Epoch
    terminal    :: Bool
end

# Maximum internal RK4 step size (s). `dt` is the control/decision interval, but a
# single RK4 step over a full `dt` is far too coarse for the LEO target (≈4%/orbit
# position error at dt=500 s). We subdivide each `dt` coast into steps no larger
# than this so integration accuracy is decoupled from the decision cadence.
const MAX_SUBSTEP = 100.0

struct SNRSDAGame{OBS,TAR,A<:Tuple} <: MG{SDAState, Tuple{Float64, Float64}}
    rk4                     ::  RK4
    epc0                    ::  Epoch
    dt                      ::  Float64
    observer                ::  OBS
    target                  ::  TAR
    discount                ::  Float64
    observer_properties     ::  ObserverProperties
    target_properties       ::  TargetProperties
    observation_conditions  ::  ObservationConditions
    altitude_bounds         ::  Tuple{Float64, Float64}
    actions                 ::  A
    function SNRSDAGame(; 
        epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0), 
        dt = 500.0,
        observer = Deterministic(sOSCtoCART(SA[R_EARTH+10_000e3, 0.5, 0.0, 0, 0, 100], use_degrees=true)),
        target = Deterministic(sOSCtoCART(SA[R_EARTH + 800e3, 0.0, 0.0, 0, 0, 0], use_degrees=true)), 
        discount = 0.98,
        n_grav = 0,
        m_grav = 0,
        drag = false,
        srp = false,
        moon = false,
        sun = false,
        relativity = false,
        observer_properties = ObserverProperties(
            0.05,     # aperture diameter: 5 cm (small-sat sensor; 20 cm made detection trivial)
            1.4,      # f-number
            9.7e-6,   # pixel size: 9.7 μm
            0.6,      # quantum efficiency
            0.5,      # dark current: 0.5 e-/pixel/s
            10.0,     # read noise: 10 e-
            2.0,      # gain: 2 e-/ADU
            0.9       # optical transmittance
        ),
        target_properties = TargetProperties(
            0.3,      # diameter: 0.3 m (small RSO; 1 m target was too bright)
            0.175,    # albedo (from paper)
            0.5       # specular fraction: 50% specular, 50% diffuse
        ),
        conditions = ObservationConditions(
            1.0,      # integration time: 1 second
            4.0,      # algorithm required SNR: 4
            1,        # binning factor: 1
            30.0,     # space background: 30 mag/arcsec² (from paper's RECONSO example)
            100       # number of background pixels: 100
        ),
        altitude_bounds = (100e3, 1e8),
        actions = (SA[-100.0, 0.0, 100.0], SA[-100.0, 0.0, 100.0])
    )
        integrator_options = (;n_grav, m_grav, drag, srp, moon, sun, relativity)
        rk4 = RK4(fderiv_earth_orbit; integrator_options...)
        return new{typeof(observer), typeof(target), typeof(actions)}(
            rk4, 
            epc0, 
            dt, 
            observer, 
            target, 
            discount, 
            observer_properties, 
            target_properties, 
            conditions,
            altitude_bounds,
            actions
        )
    end
end

MarkovGames.discount(p::SNRSDAGame) = p.discount

MarkovGames.actions(p::SNRSDAGame) = p.actions

MarkovGames.initialstate(game::SNRSDAGame)= ImplicitDistribution() do rng
    SDAState(rand(rng, game.observer), rand(rng, game.target), game.epc0, false)
end

function gen_state(game::SNRSDAGame, s::SDAState, a)
    (; observer, target, epc) = s
    (; dt) = game
    a_obs, a_tar = a
    observer_state = propagate_sat_state(game, epc, observer, a_obs)
    target_state = propagate_sat_state(game, epc, target, a_tar)
    return SDAState(observer_state, target_state, epc + dt, false)
end

altitude(x::AbstractVector) = norm(x[idx1t3], 2) - R_EARTH

function out_of_bounds(x::AbstractVector, bounds::Tuple)
    return !(bounds[1] ≤ altitude(x) ≤ bounds[2])
end

out_of_bounds(game::SNRSDAGame, x::AbstractVector) = out_of_bounds(x, game.altitude_bounds)

function MarkovGames.transition(game::SNRSDAGame, s::SDAState, a)
    sp = gen_state(game, s, a)
    # Terminate on the same step the boundary is crossed. The boundary bonus is
    # granted by the 4-arg `reward(game, s, a, sp)` on this transition, since a
    # terminal state accrues no further reward.
    terminal = out_of_bounds(game, sp.observer) || out_of_bounds(game, sp.target)
    return Deterministic(SDAState(sp.observer, sp.target, sp.epc, terminal))
end

function PhotometricSNR.calculate_snr(game::SNRSDAGame, epc, observer, target)
    snr, apparent_mag = calculate_snr(
        observer[idx1t3], observer[idx4t6],
        target[idx1t3], target[idx4t6],
        epc, game.observer_properties, game.target_properties, game.observation_conditions
    )
    return snr
end

# SNR-based reward for observing from state `s`, normalized by the detection
# threshold: r = 1 means the target is exactly at the SNR required for detection;
# r < 1 means undetectable. (Was r/100, tuned to the old over-sensitive camera
# whose SNR ran 20-8500 everywhere.) Capped at 10.
function snr_reward(game::SNRSDAGame, s::SDAState)
    (; observer, target, epc) = s
    r = can_see_sat(epc, observer, target) ? calculate_snr(game, epc, observer, target) : 0.0
    r /= game.observation_conditions.algorithm_required_snr
    return min(r, 10.0)
end

# Zero-sum vector reward: player 1 (observer) maximizes, player 2 (target) gets the
# negation. The boundary bonus is keyed on the landing state `sp` so it is collected
# exactly once, on the transition that crosses the altitude bounds (that state is
# terminal and accrues no further reward).
function MarkovGames.reward(game::SNRSDAGame, s::SDAState, a, sp::SDAState)
    r = snr_reward(game, s)
    out_of_bounds(game, sp.target)   && (r += 10.0)
    out_of_bounds(game, sp.observer) && (r -= 10.0)
    return SA[r, -r]
end

function solar_phase_angle(epc, x_obs, x_target)
    x_sun = sun_position(epc)
    xto = x_obs - x_target
    xts = x_sun - x_target
    return acos(dot(xto, xts) / (norm(xto, 2) * norm(xts, 2)))
end

MarkovGames.isterminal(::SNRSDAGame, s) = s.terminal

function propagate_sat_state(game, epc, x, a; dt=game.dt)
    (; rk4) = game
    x = SDAGames.apply_dv(x, a)
    # Apply the burn impulsively, then coast for `dt` using internal RK4 substeps no
    # larger than MAX_SUBSTEP so integration error stays small regardless of `dt`.
    n = max(1, ceil(Int, dt / MAX_SUBSTEP))
    h = dt / n
    for _ in 1:n
        x = istep(rk4, epc, h, x)
        epc += h
    end
    return x
end

function MarkovGames.convert_s(::Type{Vector{T}}, s::SDAState, game::SNRSDAGame) where T
    obs_r = s.observer[idx1t3]
    obs_v = s.observer[idx4t6]
    tar_r = s.target[idx1t3]
    tar_v = s.target[idx4t6]

    # Sun direction (unit) and, from its raw position, the solar phase angle
    # (Sun-target-observer). Encode as (sin, cos) so the network sees a smooth,
    # wrap-free representation of the angle.
    sun_raw = sun_position(s.epc)
    x_sun = normalize(sun_raw, 2)
    xto = obs_r - tar_r
    xts = sun_raw - tar_r
    cosθ = clamp(dot(xto, xts) / (norm(xto, 2) * norm(xts, 2)), -1.0, 1.0)
    sinθ = sqrt(1 - cosθ^2) # phase angle ∈ [0, π] ⇒ sin ≥ 0

    # Detectability features: visibility gate and threshold-normalized SNR (matches
    # the reward's normalization; kept uncapped here to preserve gradient at range).
    visible = can_see_sat(s.epc, s.observer, s.target)
    snr = calculate_snr(game, s.epc, s.observer, s.target)
    snr_norm = snr / game.observation_conditions.algorithm_required_snr

    # Position scaled by 1e7 m and velocity by 1e4 m/s so both land near O(1).
    return T[
        obs_r ./ 1e7;
        obs_v ./ 1e4;
        tar_r ./ 1e7;
        tar_v ./ 1e4;
        (tar_r .- obs_r) ./ 1e7;   # relative position
        (tar_v .- obs_v) ./ 1e4;   # relative velocity
        x_sun;
        sinθ; cosθ;
        visible ? one(T) : zero(T);
        snr_norm
    ]
end
