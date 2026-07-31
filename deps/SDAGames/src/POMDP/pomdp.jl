struct State
    X::Matrix{Float64} # [state_idx, sat_idx] # TODO: make static?
    t::Int
end

const Action{NO} = CartesianIndex{NO}
const Observation{NO,NTOT} = SMatrix{2, NO, Float64, NTOT}
# const Observation = Vector{Tuple{Float64, Float64}} # range, range-rate

const FLOAT_RANGE = StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
const SDAStaticSingleObsDist = MvNormal{Float64, PDMats.PDiagMat{Float64, SVector{2, Float64}}, SVector{2, Float64}}


function satellite_state(x0; epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0), dt=1.0)
    return EarthInertialState(epc0, x0; dt)
end

default_init_err(σr=1e3, σv=1.0) = [fill(σr, 3); fill(σv, 3)]

struct SDAPOMDP{NT<:NamedTuple, R<:RK4, NO, NR, NTOT} <: POMDP{State, Action{NO}, Observation{NO,NTOT}}
    ts::FLOAT_RANGE
    epcs::Vector{Epoch}
    dynamics_options::NT
    observers::NTuple{NO, Matrix{Float64}}
    rsos::NTuple{NR, DiagNormal} # NOTE: not allowing for aribtrary initial distributions
    rk4::R
    obs_noise::SVector{2,Float64}
    function SDAPOMDP(; 
            ts=0.0:1.0:6_000.0, 
            n_grav = 0,
            m_grav = 0,
            drag = false,
            srp = false,
            moon = false,
            sun = false,
            relativity = false,
            epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0), 
            x0_observers = (
                sOSCtoCART([R_EARTH + 500e3, 0.0, 90.0, 0, 0, 0], use_degrees=true),
                sOSCtoCART([R_EARTH + 1_000e3, 0.0, 0.0, 0, 0, 0], use_degrees=true)
            ), 
            x0_rso = (
                MvNormal(sOSCtoCART([R_EARTH + 500e3, 0.0, 0.0, 0, 0, 0], use_degrees=true), default_init_err()),
                MvNormal(sOSCtoCART([R_EARTH + 1_00e3, 0.0, 45.0, 0, 0, 0], use_degrees=true), default_init_err())
            ),
            obs_noise = (10.0, 0.01) # range, range_rate
        )
        epcs = map(ts) do t
            epc0 + t
        end
        dynamics_options = (;n_grav, m_grav, drag, srp, moon, sun, relativity)
        dt = step(ts)
        observers = map(x0_observers) do x0
            orb = EarthInertialState(epc0, x0; dt=dt, dynamics_options...)
            propagate_full(orb, last(ts))
        end

        rk4 = SatelliteDynamics.RK4(SatelliteDynamics.fderiv_earth_orbit, dynamics_options)
        NT = typeof(dynamics_options)
        R = typeof(rk4)
        NO = length(x0_observers)
        NTOT = 2NO
        NR = length(x0_rso)
        return new{NT, R, NO, NR, NTOT}(ts, epcs, dynamics_options, observers, x0_rso, rk4, SVector{2,Float64}(obs_noise))
    end
end

n_observers(::SDAPOMDP{NT, R, NO}) where {NT, R, NO} = NO
obstype(::SDAPOMDP{NT, R, NO, NR, NTOT}) where {NT, R, NO, NR, NTOT} = Observation{NO,NTOT}
obsmatsize(::SDAPOMDP{NT, R, NO, NR, NTOT}) where {NT, R, NO, NR, NTOT} = NTOT

vecvec2mat(x) = mapreduce(transpose, vcat, x)
vecvec2mat(x::AbstractArray{<:StaticArray}) = mapreduce(transpose∘Array, vcat, x)

Base.step(pomdp::SDAPOMDP) = step(pomdp.ts)

function POMDPs.initialstate(pomdp::SDAPOMDP)
    return ImplicitDistribution() do rng
        X = mapreduce(hcat, pomdp.rsos) do rso_dist
            rand(rng, rso_dist)
        end
        return State(X, 1)
    end
end

function POMDPs.actions(::SDAPOMDP{NT, R, NO, NR}) where {NT, R, NO, NR}
    return CartesianIndices(ntuple(Returns(1:NR), Val(NO)))
end
