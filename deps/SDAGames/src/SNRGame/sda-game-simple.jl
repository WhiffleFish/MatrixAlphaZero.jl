# ============================================================================
# SNRGameSimple
#
# A lightweight variant of `SNRSDAGame` with the *same* zero-sum SNR-observation
# idea, but a much cheaper generative function:
#
#   * State is genuinely 2D: `[x, y, vx, vy]` in the orbital plane (a dedicated
#     4D `SDAState2D`, distinct from the 6D `SDAState` used by `SNRSDAGame`).
#   * The Sun (and, implicitly, the Earth) are held at a FIXED position, so no
#     per-step ephemeris lookups (`sun_position`/`moon_position`) are needed.
#   * Trajectories are propagated with the closed-form two-body (Kepler) solution
#     in the plane instead of substepped RK4 over a full force model.
#
# Because the Sun is static and the Moon is dropped, the SNR / visibility logic
# is re-derived here to take the fixed Sun position as an argument rather than
# recomputing it from the epoch.
# ============================================================================

# 2D Cartesian state: position (x, y) in indices 1:2, velocity (vx, vy) in 3:4.
const StateVec2D = SVector{4, Float64}
const pos2 = StaticArrays.SUnitRange(1, 2)
const vel2 = StaticArrays.SUnitRange(3, 4)

struct SDAState2D
    observer    :: StateVec2D
    target      :: StateVec2D
    epc         :: Epoch
    terminal    :: Bool
end

# Project a 6D Cartesian ECI state to the in-plane 2D state. Exact for the
# equatorial (zero-inclination) orbits used here, where z = vz = 0.
cart_to_2d(x) = StateVec2D(x[1], x[2], x[4], x[5])

"""
    sOSCtoCART2D(x_oe; use_degrees=false, μ=GM_EARTH) -> StateVec2D

2D analogue of `SatelliteDynamics.sOSCtoCART` for the planar dynamics used by
`SNRGameSimple`. Converts osculating orbital elements to the Cartesian state
`[x, y, vx, vy]` (a `StateVec2D`).

Since the model is planar, the two out-of-plane elements of the usual 6-element
set — inclination `i` and RAAN `Ω` — don't exist; every orbit is implicitly
equatorial (`i = 0`). The remaining four elements occupy `x_oe` in the same
relative order `sOSCtoCART` uses (dropping elements 3 and 4):

| index | element | symbol | units   | description                                        |
|:-----:|---------|:------:|---------|-----------------------------------------------------|
| 1     | semi-major axis      | `a` | m   | size of the orbit                                    |
| 2     | eccentricity         | `e` | –   | shape of the orbit (`0 ≤ e < 1`; hyperbolic/parabolic orbits are not supported) |
| 3     | argument of periapsis| `ω` | rad | angle from the fixed +x axis to periapsis, in-plane  |
| 4     | mean anomaly         | `M` | rad | position of the body along the orbit at epoch        |

Pass `use_degrees=true` to give/receive `ω` and `M` in degrees (matching
`sOSCtoCART`'s convention); `a` and `e` are always in meters/dimensionless.

Equivalent (to machine precision) to
`cart_to_2d(sOSCtoCART(SA[a, e, 0.0, 0.0, ω, M]; use_degrees))`, but does not
require constructing or projecting a 6D state.
"""
function sOSCtoCART2D(x_oe::AbstractVector; use_degrees::Bool = false, μ::Real = GM_EARTH)
    a, e = x_oe[1], x_oe[2]
    ω, M = x_oe[3], x_oe[4]
    if use_degrees
        ω = deg2rad(ω)
        M = deg2rad(M)
    end
    0 ≤ e < 1 || throw(ArgumentError("sOSCtoCART2D only supports elliptic orbits (0 ≤ e < 1); got e = $e"))

    # Solve Kepler's equation M = E - e sin(E) for the eccentric anomaly E via
    # Newton's method.
    E = M
    for _ in 1:100
        ΔE = (E - e * sin(E) - M) / (1 - e * cos(E))
        E -= ΔE
        abs(ΔE) < 1e-13 && break
    end

    # True anomaly from eccentric anomaly.
    ν = 2 * atan(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2))

    r = a * (1 - e * cos(E))
    p = a * (1 - e^2)

    # Position/velocity in the perifocal frame (periapsis along local +x).
    r_pf = r .* SA[cos(ν), sin(ν)]
    v_pf = sqrt(μ / p) .* SA[-sin(ν), e + cos(ν)]

    # Rotate from the perifocal frame into the fixed 2D frame by ω.
    cω, sω = cos(ω), sin(ω)
    R = SA[cω -sω; sω cω]
    r_xy = R * r_pf
    v_xy = R * v_pf

    return StateVec2D(r_xy[1], r_xy[2], v_xy[1], v_xy[2])
end

# ---------------------------------------------------------------------------
# Closed-form two-body (Kepler) propagation via universal variables.
# Works for any conic and any spatial dimension (only dot/norm are used); for
# the bound (elliptic) orbits used here it matches a fine-step two-body RK4
# integration to within nanometres.
# ---------------------------------------------------------------------------

# Stumpff functions C(z), S(z).
stumpff_C(z) = z > 0 ? (1 - cos(sqrt(z))) / z :
               (z < 0 ? (cosh(sqrt(-z)) - 1) / (-z) : 0.5)

function stumpff_S(z)
    if z > 0
        s = sqrt(z)
        return (s - sin(s)) / s^3
    elseif z < 0
        s = sqrt(-z)
        return (sinh(s) - s) / s^3
    else
        return 1 / 6
    end
end

"""
    kepler_propagate(r0, v0, dt, μ)

Propagate a Cartesian state `(r0, v0)` forward by `dt` seconds under two-body
gravity with parameter `μ`, using the universal-variable formulation and
Lagrange `f`/`g` coefficients. Returns `(r, v)`. Dimension-agnostic: a 2D input
yields a 2D output.
"""
function kepler_propagate(r0, v0, dt, μ; tol = 1e-9, maxiter = 300)
    r0m = norm(r0)
    v0m = norm(v0)
    vr0 = dot(r0, v0) / r0m
    α   = 2 / r0m - v0m^2 / μ           # reciprocal of the semi-major axis (1/a)

    √μ = sqrt(μ)
    χ  = √μ * abs(α) * dt               # universal anomaly, initial guess
    for _ in 1:maxiter
        z = α * χ^2
        C = stumpff_C(z)
        S = stumpff_S(z)
        F  = r0m * vr0 / √μ * χ^2 * C + (1 - α * r0m) * χ^3 * S + r0m * χ - √μ * dt
        dF = r0m * vr0 / √μ * χ * (1 - α * χ^2 * S) + (1 - α * r0m) * χ^2 * C + r0m
        ratio = F / dF
        χ -= ratio
        abs(ratio) < tol && break
    end

    z = α * χ^2
    C = stumpff_C(z)
    S = stumpff_S(z)
    f = 1 - χ^2 / r0m * C
    g = dt - χ^3 / √μ * S
    r = f * r0 + g * v0
    rm = norm(r)
    ḟ = √μ / (rm * r0m) * (α * χ^3 * S - χ)
    ġ = 1 - χ^2 / rm * C
    v = ḟ * r0 + ġ * v0
    return r, v
end

# Impulsive prograde burn in 2D: add Δv along the current velocity direction.
function apply_dv_2d(x, Δv)
    Δvp = first(Δv)
    v = x[vel2]
    v̂ = normalize(v, 2)
    return StateVec2D(vcat(x[pos2], v .+ Δvp .* v̂))
end

# Fixed Sun position, in the orbital plane at 1 AU along +x.
const AU_METERS = 1.495978707e11
const DEFAULT_SUN_POSITION = SA[AU_METERS, 0.0]

struct SNRGameSimple{OBS,TAR,A<:Tuple} <: MG{SDAState2D, Tuple{Float64, Float64}}
    epc0                    ::  Epoch
    dt                      ::  Float64
    μ                       ::  Float64
    sun_position            ::  SVector{2, Float64}
    observer                ::  OBS
    target                  ::  TAR
    discount                ::  Float64
    observer_properties     ::  ObserverProperties
    target_properties       ::  TargetProperties
    observation_conditions  ::  ObservationConditions
    altitude_bounds         ::  Tuple{Float64, Float64}
    actions                 ::  A
    function SNRGameSimple(;
        epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0),
        dt = 500.0,
        μ = GM_EARTH,
        sun_position = DEFAULT_SUN_POSITION,
        observer = Deterministic(sOSCtoCART2D(SA[R_EARTH+10_000e3, 0.5, 0.0, 100], use_degrees=true)),
        target = Deterministic(sOSCtoCART2D(SA[R_EARTH + 800e3, 0.0, 0.0, 0], use_degrees=true)),
        discount = 0.98,
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
        return new{typeof(observer), typeof(target), typeof(actions)}(
            epc0,
            dt,
            μ,
            SVector{2, Float64}(sun_position),
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

MarkovGames.discount(p::SNRGameSimple) = p.discount

MarkovGames.actions(p::SNRGameSimple) = p.actions

MarkovGames.initialstate(game::SNRGameSimple) = ImplicitDistribution() do rng
    SDAState2D(rand(rng, game.observer), rand(rng, game.target), game.epc0, false)
end

MarkovGames.isterminal(::SNRGameSimple, s) = s.terminal

# ---------------------------------------------------------------------------
# Dynamics: impulsive burn + closed-form two-body coast (no integration loop).
# ---------------------------------------------------------------------------

function propagate_sat_state(game::SNRGameSimple, x, a; dt = game.dt)
    x = apply_dv_2d(x, a)
    r, v = kepler_propagate(x[pos2], x[vel2], dt, game.μ)
    return StateVec2D(vcat(r, v))
end

function gen_state(game::SNRGameSimple, s::SDAState2D, a)
    (; observer, target, epc) = s
    a_obs, a_tar = a
    observer_state = propagate_sat_state(game, observer, a_obs)
    target_state   = propagate_sat_state(game, target, a_tar)
    return SDAState2D(observer_state, target_state, epc + game.dt, false)
end

altitude2d(x::AbstractVector) = norm(x[pos2], 2) - R_EARTH
out_of_bounds2d(x, bounds) = !(bounds[1] ≤ altitude2d(x) ≤ bounds[2])
out_of_bounds(game::SNRGameSimple, x::AbstractVector) = out_of_bounds2d(x, game.altitude_bounds)

function MarkovGames.transition(game::SNRGameSimple, s::SDAState2D, a)
    sp = gen_state(game, s, a)
    # Terminate on the same step the boundary is crossed; the boundary bonus is
    # granted by the 4-arg reward on this transition (a terminal state accrues no
    # further reward).
    terminal = out_of_bounds(game, sp.observer) || out_of_bounds(game, sp.target)
    return Deterministic(SDAState2D(sp.observer, sp.target, sp.epc, terminal))
end

# ---------------------------------------------------------------------------
# Static-Sun photometry / visibility.
#
# The heavy per-frame flux, background and pixel-count terms are reused verbatim
# from `PhotometricSNR` (they only use dot/norm, so 2D vectors are fine); only
# the solar-geometry pieces are re-derived here to take the fixed Sun position
# instead of an epoch-driven ephemeris.
# ---------------------------------------------------------------------------

# Solar phase angle (Sun–target–observer) using a fixed Sun position.
function solar_phase_angle_static(x_sun, observer_pos, target_pos)
    t2o = normalize(observer_pos - target_pos, 2)
    t2s = normalize(x_sun - target_pos, 2)
    return acos(clamp(dot(t2o, t2s), -1.0, 1.0))
end

# Apparent visual magnitude of the target (paper Eq. A.2) with a fixed Sun.
function apparent_magnitude_static(x_sun, observer_pos, target_pos, target_props)
    range = norm(observer_pos - target_pos, 2)
    ψ = solar_phase_angle_static(x_sun, observer_pos, target_pos)
    ρ = PhotometricSNR.calculate_reflectivity(ψ, target_props.specular_fraction)
    area = π * (target_props.diameter / 2)^2
    m_v_sun = -26.7
    return m_v_sun - 2.5 * log10(target_props.albedo * ρ * area / range^2)
end

# SNR (paper's CCD equation, A.29) evaluated against a fixed Sun position.
function calculate_snr_static(x_sun, observer, target, observer_props, target_props, obs_conditions)
    o_pos, o_vel = observer[pos2], observer[vel2]
    t_pos, t_vel = target[pos2],   target[vel2]

    apparent_mag = apparent_magnitude_static(x_sun, o_pos, t_pos, target_props)
    q_target = PhotometricSNR.calculate_photon_flux(apparent_mag, observer_props)
    q_p_sky  = PhotometricSNR.calculate_background_flux(observer_props, obs_conditions)
    q_p_dark = observer_props.dark_current
    m = PhotometricSNR.calculate_pixel_count(observer_props, o_pos, o_vel, t_pos, t_vel, obs_conditions.integration_time)

    t   = obs_conditions.integration_time
    z   = obs_conditions.num_background_pixels
    n   = obs_conditions.binning_factor
    σ_r = observer_props.read_noise

    numerator   = q_target * t
    denominator = sqrt(q_target * t + m * (q_p_sky + q_p_dark) * t + (1 + m / z) * σ_r^2 / n^2)
    return numerator / denominator
end

PhotometricSNR.calculate_snr(game::SNRGameSimple, s::SDAState2D) = calculate_snr_static(
    game.sun_position, s.observer, s.target,
    game.observer_properties, game.target_properties, game.observation_conditions
)

# Is the target visible? Same geometry as `can_see_sat` but with a fixed Sun and
# no Moon: not earth-occluded, not looking into the Sun, and sunlit (not in the
# Earth's shadow). All helpers are dimension-agnostic, so 2D positions work.
function can_see_sat_static(x_sun, x_obs, x_sat)
    length(x_obs) == 4 && (x_obs = x_obs[pos2])
    length(x_sat) == 4 && (x_sat = x_sat[pos2])
    return !(
        earth_occlusion(x_obs, x_sat) ||
        looking_into_sun(x_sun, x_obs, x_sat) ||
        !is_sat_sunlit_static(x_sun, x_sat)
    )
end

# Sunlit test: target is outside the Earth's (cylindrical) shadow. Mirrors the
# Earth branch of `is_sat_backlit`, dropping the Moon.
function is_sat_sunlit_static(x_sun, x_obj)
    v_se = -x_sun
    v_proj = (dot(v_se, x_obj) / dot(v_se, v_se)) * v_se
    v_perp = x_obj - v_proj
    return norm(v_perp, 2) ≥ R_EARTH
end

# ---------------------------------------------------------------------------
# Reward: identical shaping to `SNRSDAGame`, threshold-normalized and capped.
# ---------------------------------------------------------------------------

function snr_reward(game::SNRGameSimple, s::SDAState2D)
    r = can_see_sat_static(game.sun_position, s.observer, s.target) ?
        calculate_snr(game, s) : 0.0
    r /= game.observation_conditions.algorithm_required_snr
    return min(r, 10.0)
end

function MarkovGames.reward(game::SNRGameSimple, s::SDAState2D, a, sp::SDAState2D)
    r = snr_reward(game, s)
    out_of_bounds(game, sp.target)   && (r += 10.0)
    out_of_bounds(game, sp.observer) && (r -= 10.0)
    return SA[r, -r]
end

# ---------------------------------------------------------------------------
# Neural-network state encoding: same features as `SNRSDAGame` but in 2D, and
# the Sun direction / phase angle come from the fixed Sun position.
# ---------------------------------------------------------------------------

function MarkovGames.convert_s(::Type{Vector{T}}, s::SDAState2D, game::SNRGameSimple) where T
    obs_r = s.observer[pos2]
    obs_v = s.observer[vel2]
    tar_r = s.target[pos2]
    tar_v = s.target[vel2]

    # Solar phase angle (Sun-target-observer), encoded as (sin, cos) so the
    # network sees a smooth, wrap-free angle. The Sun *direction* itself isn't
    # included as a feature: it's fixed for the lifetime of the game, so it
    # carries no information the network could act on.
    xto = obs_r - tar_r
    xts = game.sun_position - tar_r
    cosθ = clamp(dot(xto, xts) / (norm(xto, 2) * norm(xts, 2)), -1.0, 1.0)
    sinθ = sqrt(1 - cosθ^2) # phase angle ∈ [0, π] ⇒ sin ≥ 0

    # Detectability features: visibility gate and threshold-normalized SNR
    # (matches the reward's normalization; kept uncapped here to preserve
    # gradient at range).
    visible = can_see_sat_static(game.sun_position, s.observer, s.target)
    snr = calculate_snr(game, s)
    snr_norm = snr / game.observation_conditions.algorithm_required_snr

    # Position scaled by 1e7 m and velocity by 1e4 m/s so both land near O(1).
    return T[
        obs_r ./ 1e7;
        obs_v ./ 1e4;
        tar_r ./ 1e7;
        tar_v ./ 1e4;
        (tar_r .- obs_r) ./ 1e7;   # relative position
        (tar_v .- obs_v) ./ 1e4;   # relative velocity
        sinθ; cosθ;
        visible ? one(T) : zero(T);
        snr_norm
    ]
end
