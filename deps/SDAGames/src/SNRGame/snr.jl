module PhotometricSNR

export calculate_snr, ObserverProperties, TargetProperties, ObservationConditions

using LinearAlgebra
using SatelliteDynamics
using Dates

# Constants
const PLANCK_CONSTANT = 6.62607015e-34  # J·s
const SPEED_OF_LIGHT = 299792458.0      # m/s
const BOLTZMANN_CONSTANT = 1.380649e-23 # J/K
const SUN_RADIUS = 696340000.0          # m
const AU = 149597870700.0               # m (1 Astronomical Unit)
const SUN_TEMPERATURE = 5778.0          # K
const MAG_0_FLUX = 5.6e10               # photons/s/m² (from paper)

# Struct to hold observer satellite properties
struct ObserverProperties
    aperture_diameter       ::  Float64 # m
    f_number                ::  Float64
    pixel_size              ::  Float64 # m
    quantum_efficiency      ::  Float64
    dark_current            ::  Float64 # e-/pixel/s
    read_noise              ::  Float64 # e-
    gain                    ::  Float64 # e-/ADU
    optical_transmittance   ::  Float64
end

# Struct to hold target satellite properties
struct TargetProperties
    diameter            ::  Float64 # m
    albedo              ::  Float64
    specular_fraction   ::  Float64 # Fraction of reflectivity that's specular vs diffuse
end

# Struct to hold observation conditions
struct ObservationConditions
    integration_time        ::  Float64 # s
    algorithm_required_snr  ::  Float64
    binning_factor          ::  Int
    background_magnitude    ::  Float64 # mag/arcsec²
    num_background_pixels   ::  Int     # number of pixels used for background estimation
end

"""
    calculate_sun_direction(time::DateTime)

Calculate unit vector from Earth to Sun for a given time.
This is a simplified model for demonstration.
"""
function calculate_sun_direction(time::DateTime)
    # Very simplified Earth-Sun direction model
    # For a proper implementation, use an ephemeris package
    # This just creates a unit vector that moves around the ecliptic plane
    day_of_year = Dates.dayofyear(time)
    year_fraction = day_of_year / (Dates.isleapyear(time) ? 366.0 : 365.0)
    angle = 2π * year_fraction
    
    # Unit vector pointing from Earth to Sun (in ECI frame)
    return [cos(angle), sin(angle), 0.0]
end

calculate_sun_direction(epc::Epoch) = normalize(sun_position(epc), 2)

angle_between(a,b) = acos(dot(a,b) / (norm(a,2) * norm(b,2)))

"""
    calculate_solar_phase_angle(observer_position, target_position, time)

Calculate the solar phase angle (angle between sun-target-observer)
"""
function calculate_solar_phase_angle(observer_position, target_position, time)
    # Vector from target to observer
    target_to_observer = normalize(observer_position - target_position)
    
    # Vector from target to sun
    sun_direction = calculate_sun_direction(time)
    target_to_sun = sun_direction
    
    # Calculate the solar phase angle (angle between these vectors)
    phase_angle = angle_between(target_to_observer, target_to_sun)
    # phase_angle = acos(clamp(dot(target_to_observer, target_to_sun), -1.0, 1.0))
    
    return phase_angle
end

"""
    calculate_reflectivity(phase_angle, specular_fraction)

Calculate the combined specular and diffuse reflectivity components
based on Equations 24 and 25 in the paper.
"""
function calculate_reflectivity(phase_angle, specular_fraction)
    # Equation 24: specular component
    ρ_spec = 1.0 / (4π)
    
    # Equation 25: diffuse component
    ψ = phase_angle
    ρ_diff = (2/3π) * (sin(ψ) + (π - ψ) * cos(ψ))
    
    # Combine based on the specified fraction
    ρ = specular_fraction * ρ_spec + (1 - specular_fraction) * ρ_diff
    
    return ρ
end

"""
    calculate_apparent_magnitude(observer_pos, target_pos, target_props, time)

Calculate the apparent visual magnitude of the target satellite.
Implements equation A.2 from the paper.
"""
function calculate_apparent_magnitude(observer_pos, target_pos, target_props, time)
    # Calculate range to target
    range = norm(observer_pos - target_pos)
    
    # Calculate solar phase angle
    phase_angle = calculate_solar_phase_angle(observer_pos, target_pos, time)
    
    # Calculate reflectivity
    ρ = calculate_reflectivity(phase_angle, target_props.specular_fraction)
    
    # Calculate cross-sectional area (assuming spherical target)
    area = π * (target_props.diameter / 2)^2
    
    # Calculate apparent magnitude using equation A.2
    # Note: m_v,⊙ is the visual magnitude of the Sun, typically -26.7
    m_v_sun = -26.7
    
    # Simplified version of Equation A.2
    m_v = m_v_sun - 2.5 * log10(target_props.albedo * ρ * area / range^2)
    
    return m_v
end

"""
    calculate_photon_flux(apparent_magnitude, observer_props)

Calculate the photon flux from the target satellite reaching the detector.
Implements equations A.3 and A.8 from the paper.
"""
function calculate_photon_flux(apparent_magnitude, observer_props)
    # Equation A.3: Convert magnitude to photon flux density
    Φ_target = MAG_0_FLUX * exp10(-0.4 * apparent_magnitude)
    
    # Equation A.8: Calculate photon flux captured by optical system
    # For space-based observation, τ_atm = 1.0 (no atmosphere)
    D = observer_props.aperture_diameter
    QE = observer_props.quantum_efficiency
    τ_opt = observer_props.optical_transmittance
    
    q_target = Φ_target * (π * D^2 / 4) * QE * τ_opt
    
    return q_target
end

"""
    calculate_background_flux(observer_props, obs_conditions)

Calculate the background sky photon flux per pixel.
Implements equations A.9 through A.16 from the paper.
"""
function calculate_background_flux(observer_props, obs_conditions)
    # Convert background magnitude to photon radiance (similar to Eq. A.9)
    I_sky = obs_conditions.background_magnitude
    L_sky = MAG_0_FLUX * 10^(-0.4 * I_sky) * (π/180)^2 / 3600^2
    
    # Calculate IFOV (instantaneous field of view) of a pixel (Eq. A.11)
    p = observer_props.pixel_size
    N = observer_props.f_number
    D = observer_props.aperture_diameter
    IFOV = p / (N * D)
    
    # Calculate background photon flux per pixel (simplified from Eqs. A.13 to A.16)
    # For space observation, use Equation A.16 directly
    # g in the paper is the optical throughput
    g = π/4 * (1/N)^2
    q_p_sky = L_sky * g * observer_props.quantum_efficiency * observer_props.optical_transmittance * p^2
    
    return q_p_sky
end

"""
    calculate_angular_velocity(observer_pos, observer_vel, target_pos, target_vel)

Calculate the angular velocity of the target as seen from the observer.
Returns angular velocity in radians per second.
"""
function calculate_angular_velocity(observer_pos, observer_vel, target_pos, target_vel)
    # Calculate relative position vector
    r = target_pos - observer_pos
    r_norm = norm(r)
    
    # Calculate relative velocity vector
    v_rel = target_vel - observer_vel
    
    # Calculate radial velocity component (dot product of normalized position and velocity)
    r_unit = r / r_norm
    v_radial = dot(r_unit, v_rel)
    
    # Calculate transverse (perpendicular) velocity component
    v_transverse = norm(v_rel - v_radial * r_unit)
    
    # Angular velocity ω = v_transverse / r
    # This is the rate of change of the angle in the sky (radians per second)
    ω = v_transverse / r_norm
    
    return ω
end

"""
    calculate_pixel_count(observer_props, observer_pos, observer_vel, target_pos, target_vel, integration_time)

Calculate the number of pixels the target occupies on the sensor.
Implements equations A.31 through A.33 from the paper.
"""
function calculate_pixel_count(observer_props, observer_pos, observer_vel, target_pos, target_vel, integration_time)
    # Calculate IFOV
    p = observer_props.pixel_size
    N = observer_props.f_number
    D = observer_props.aperture_diameter
    IFOV = p / (N * D)
    
    # Calculate PSF due to diffraction (Equation A.30)
    # Using a typical wavelength of visible light (550 nm)
    λ = 550e-9
    θ_A = 2.44 * λ / D
    
    # For space-based observation, no atmospheric seeing, so θ_S = 0
    θ_S = 0.0
    
    # Determine the point spread function size (Equation A.31)
    θ = max(θ_S, θ_A, IFOV)
    
    # Calculate initial pixel count (Equation A.32)
    m_i = θ == IFOV ? 1.0 : π * (θ * N * D / p)^2 / 4
    
    # Calculate angular velocity properly
    ω = calculate_angular_velocity(observer_pos, observer_vel, target_pos, target_vel)
    
    # Calculate the number of pixels due to target motion (Equation A.33)
    m = m_i + (m_i * ω * integration_time / IFOV)
    
    # Ensure at least one pixel
    return max(1.0, m)
end

"""
    calculate_snr(observer_pos, observer_vel, target_pos, target_vel, time, 
                 observer_props, target_props, obs_conditions)

Calculate the Signal-to-Noise Ratio for satellite observation.
Implements equation A.29 (the CCD Equation) from the paper.
"""
function calculate_snr(
    observer_pos    ::  AbstractVector{Float64},
    observer_vel    ::  AbstractVector{Float64},
    target_pos      ::  AbstractVector{Float64},
    target_vel      ::  AbstractVector{Float64},
    time            ::  Union{DateTime, Epoch},
    observer_props  ::  ObserverProperties,
    target_props    ::  TargetProperties,
    obs_conditions  ::  ObservationConditions
)
    # 1. Calculate apparent magnitude of target
    apparent_mag = calculate_apparent_magnitude(observer_pos, target_pos, target_props, time)
    
    # 2. Calculate target photon flux
    q_target = calculate_photon_flux(apparent_mag, observer_props)
    
    # 3. Calculate background flux per pixel
    q_p_sky = calculate_background_flux(observer_props, obs_conditions)
    
    # 4. Get dark current per pixel
    q_p_dark = observer_props.dark_current
    
    # 5. Calculate number of pixels the target occupies
    m = calculate_pixel_count(observer_props, observer_pos, observer_vel, target_pos, target_vel, obs_conditions.integration_time)
    
    # 6. Calculate SNR using equation A.29 (the CCD Equation)
    t = obs_conditions.integration_time
    z = obs_conditions.num_background_pixels
    n = obs_conditions.binning_factor
    σ_r = observer_props.read_noise
    
    numerator = q_target * t
    denominator = sqrt(q_target * t + m * (q_p_sky + q_p_dark) * t + (1 + m/z) * σ_r^2/n^2)
    
    snr = numerator / denominator
    
    return snr, apparent_mag
end

# Example usage
function example()
    # Setup observer satellite properties
    observer = ObserverProperties(
        0.2,      # aperture diameter: 20 cm
        1.4,      # f-number
        9.7e-6,   # pixel size: 9.7 μm
        0.6,      # quantum efficiency
        0.5,      # dark current: 0.5 e-/pixel/s
        10.0,     # read noise: 10 e-
        2.0,      # gain: 2 e-/ADU
        0.9       # optical transmittance
    )
    
    # Setup target satellite properties
    target = TargetProperties(
        1.0,      # diameter: 1 meter
        0.175,    # albedo (from paper)
        0.5       # specular fraction: 50% specular, 50% diffuse
    )
    
    # Setup observation conditions
    conditions = ObservationConditions(
        1.0,      # integration time: 1 second
        4.0,      # algorithm required SNR: 4
        1,        # binning factor: 1
        30.0,     # space background: 30 mag/arcsec² (from paper's RECONSO example)
        100       # number of background pixels: 100
    )
    
    # Example state vectors (ECI coordinates in meters and m/s)
    observer_pos = [7000e3, 0.0, 0.0]  # Observer at 7000 km altitude
    observer_vel = [0.0, 7.5e3, 0.0]   # Observer velocity 7.5 km/s
    target_pos = [7100e3, 100e3, 0.0]  # Target at different position
    target_vel = [0.0, 7.3e3, 0.0]     # Target with slightly different velocity
    
    # Current time
    current_time = Dates.now()
    
    # Calculate SNR
    snr, apparent_mag = calculate_snr(
        observer_pos, observer_vel, target_pos, target_vel, 
        current_time, observer, target, conditions
    )
    
    println("Target apparent magnitude: ", apparent_mag)
    println("Calculated SNR: ", snr)
    println("Detection possible: ", snr >= conditions.algorithm_required_snr)
    
    return snr
end

end # module
