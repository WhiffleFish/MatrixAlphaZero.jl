const CORE_DISTRIBUTION_NAME =
    "correlated_leo_target600_1200km_delta300km_separation10_120deg"
const CORE_TARGET_ALTITUDE_BOUNDS = (600e3, 1_200e3)
const CORE_RELATIVE_ALTITUDE_BOUNDS = (-300e3, 300e3)
const CORE_ABS_PHASE_SEPARATION_BOUNDS = deg2rad.((10.0, 120.0))

"""
    core_initialstate_distribution(game)

Joint in-distribution initial-state law for the July 21 SDA experiment.

The target is sampled in LEO. The observer altitude and phase are sampled
relative to that target, so this is deliberately not representable as the
product of independent observer and target marginals.
"""
function core_initialstate_distribution(game)
    return ImplicitDistribution() do rng
        target_altitude = rand(
            rng,
            Distributions.Uniform(CORE_TARGET_ALTITUDE_BOUNDS...),
        )
        relative_altitude = rand(
            rng,
            Distributions.Uniform(CORE_RELATIVE_ALTITUDE_BOUNDS...),
        )
        target_phase = 2π * rand(rng)
        abs_separation = rand(
            rng,
            Distributions.Uniform(CORE_ABS_PHASE_SEPARATION_BOUNDS...),
        )
        signed_separation = rand(rng, Bool) ? abs_separation : -abs_separation
        observer_phase = mod2pi(target_phase + signed_separation)

        observer = SNRGame.sOSCtoCART2D([
            R_EARTH + target_altitude + relative_altitude,
            0.0,
            0.0,
            observer_phase,
        ])
        target = SNRGame.sOSCtoCART2D([
            R_EARTH + target_altitude,
            0.0,
            0.0,
            target_phase,
        ])
        return SNRGame.SDAState2D(observer, target, game.epc0, false)
    end
end
