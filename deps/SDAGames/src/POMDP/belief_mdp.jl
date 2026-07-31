const AbstractFullNormal{M<:AbstractMatrix,V<:AbstractVector} = MvNormal{Float64, PDMat{Float64, M}, V}

struct HybridBelief{C<:AbstractVector{<:AbstractFullNormal}}
    discrete::SparseCat{UnitRange{Int64}, Vector{Float64}}
    continuous::C
end

struct BMDPState{B<:AbstractFullNormal}
    t::Int
    b::HybridBelief{Vector{B}}
end

const BMDPAction = Int

const BMDPObservation = SVector{2,Float64}

struct ObjectOfInterest{MV<:MvNormal, T<:Tuple}
    x::MV
    integrators::T
    function ObjectOfInterest(x::MvNormal, params) # params is iterable of named tuple
        integrators = map(
            p->RK4(SatelliteDynamics.fderiv_earth_orbit, merge(DEFAULT_INTEGRATOR_OPTIONS, p)), 
            params
        )
        return new{typeof(x), typeof(integrators)}(x, integrators)
    end
end

ObjectOfInterest(μ::AbstractVector, Σ::AbstractMatrix, params) = ObjectOfInterest(MvNormal(μ, Σ), params)
ObjectOfInterest(μ::AbstractVector, Σ::AbstractVector, params) = ObjectOfInterest(MvNormal(μ, diagm(Σ)), params)

struct SpaceObject{MV<:MvNormal, T<:RK4}
    x::MV
    rk4::T
    function SpaceObject(x::MvNormal; kwargs...)
        rk4 = RK4(SatelliteDynamics.fderiv_earth_orbit, merge(DEFAULT_INTEGRATOR_OPTIONS, kwargs))
        return new{typeof(x), typeof(rk4)}(x,rk4)
    end
end

SpaceObject(x::AbstractVector; kwargs...) = SpaceObject(MvNormal(x, default_init_err()); kwargs...)

struct SDABMDP{NO,OBS<:ObjectOfInterest,UKF<:UnscentedKalmanFilter} <: MDP{BMDPState, BMDPAction}
    ts::FLOAT_RANGE
    epcs::Vector{Epoch}
    observers::NTuple{NO, Matrix{Float64}}
    object::OBS
    obs_noise::SVector{2,Float64}
    obs_plan::BitArray{3}
    kf::UKF
    w_belief::Float64
    rBelFun::String
    discount::Float64
    trueModel::Int
    function SDABMDP(; 
            ts=0.0:20.0:900,    #6_000.0,
            epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0), 
            observers = (
                Observer(
                    sOSCtoCART([R_EARTH, 0.0, 0.0, 0, 0, 0], use_degrees=true);
                    ground_based = true
                ),
                Observer(
                    sOSCtoCART([R_EARTH, 0.0, 0.0, 0, 0, 360/3], use_degrees=true);
                    ground_based = true
                ),
                Observer(
                    sOSCtoCART([R_EARTH, 0.0, 0.0, 0, 0, 2*360/3], use_degrees=true);
                    ground_based = true
                )
            ),
            object_of_interest = ObjectOfInterest(
                sOSCtoCART(SA[R_EARTH + 500e3, 0.0, 0.0, 0, 0, 0], use_degrees=true), 
                SA[1,1,1,1e-3,1e-3,1e-3],
                (
                    (; ),
                    (; drag=true, area_drag=1.0),
                    (; drag=true, area_drag=5.0)
                )
            ),
            objects = [
                SpaceObject(sOSCtoCART([R_EARTH + 5000e3, 0.0, 0.0, 0, 0, 135], use_degrees=true)),
                SpaceObject(sOSCtoCART([R_EARTH + 10_000e3, 0.0, 0.0, 0, 0, 180], use_degrees=true)),
                SpaceObject(sOSCtoCART([R_EARTH + 10_000e3, 0.0, 0.0, 0, 0, 270], use_degrees=true)),
                SpaceObject(sOSCtoCART([R_EARTH + 2_000e3, 0.0, 0.0, 0, 0, 270], use_degrees=true)),
                # SpaceObject(sOSCtoCART([R_EARTH + 50_000e3, 0.0, 0.0, 0, 0, 270], use_degrees=true)),
                # SpaceObject(sOSCtoCART([R_EARTH + 49_000e3, 0.0, 0.0, 0, 0, 270], use_degrees=true)),
                # SpaceObject(sOSCtoCART([R_EARTH + 40_000e3, 0.0, 0.0, 0, 0, 270], use_degrees=true))
            ],
            obs_noise = (10.0, 0.01), # range, range_rate
            ukf = UnscentedKalmanFilter(dynamics, measurement, PDMat(diagm(@SArray(ones(6))*1e-6)), PDMat(SA[obs_noise[1] 0; 0 obs_noise[2]]), ny=2, nu=1),
            w_belief = 1.0,
            rBelFun = "entropy",
            discount = 0.95,
            trueModel = 1
        )
        epcs = map(ts) do t
            epc0 + t
        end
        dt = step(ts)
        X_obs = map(observers) do obs
            orb = EarthInertialState(obs.rk4, dt, epc0, obs.x, nothing)
            obs.ground_based ? propagate_full_ground(orb, last(ts)) : propagate_full(orb, last(ts))
        end
        X_obj = map(objects) do obj
            orb = EarthInertialState(obj.rk4, dt, epc0, mean(obj.x), nothing)
            SDAGames.propagate_full(orb, last(ts))
        end
        NO = length(observers)
        obs_plan = observation_plan(X_obs, X_obj)

        return new{NO, typeof(object_of_interest), typeof(ukf)}(ts, epcs, X_obs, object_of_interest, SVector{2,Float64}(obs_noise), obs_plan, ukf, w_belief, rBelFun, discount,trueModel)
    end
end

function observation_plan(X_obs, X_obj; kwargs...)
    O = SDAGames.occlusion_matrix(X_obs, X_obj)
    return SDAGames.observation_maximin_occlusion_ilp(O, kwargs...)
end

# TODO: make actions occlusion dependent
POMDPs.actions(p::SDABMDP) = 0:length(p.observers)

function POMDPs.actions(p::SDABMDP, s::BMDPState)
    a = [0]
    t = s.t
    hyp = rand(s.b.discrete)
    x = [mean(s.b.continuous[hyp])]
    X_obs = map(mat -> mat[:,t:t], p.observers)
    O = SDAGames.occlusion_matrix(X_obs, x, 1)

    for i in findall(O[:,:,1].==1)
        push!(a,LinearIndices(O[:,:,1])[i])
    end

    return a
end

POMDPs.obstype(::SDABMDP) = BMDPObservation

POMDPs.discount(p::SDABMDP) = p.discount

POMDPs.isterminal(mdp::SDABMDP, s::BMDPState) = s.t ≥ lastindex(mdp.ts)

propagation_function(rk4::RK4, epc::Epoch, dt::Float64) = x -> istep(rk4, epc, dt, x)

function dynamics(x, u, p, t=0)
    (; rk4, epc, dt) = p
    return istep(rk4, epc, dt, x)
end

measurement(x_obj, u, p, t=0) = _measurement(x_obj, p.x_obs)

function _measurement(x_obj, x_obs)
    Rs = x_obs[1:3]
    Vs = x_obs[4:6]
    R = x_obj[1:3]
    V = x_obj[4:6]
    return SA[
        station_range(R, Rs, V, Vs),
        station_range_rate(R, Rs, V, Vs)
    ]
end

function POMDPs.gen(mdp::SDABMDP, s::BMDPState, a::Int, rng::AbstractRNG)
    sp = gen_state(mdp, s, a, rng)
    return (;sp)
end

function POMDPs.initialstate(mdp::SDABMDP)
    n_discrete  = length(mdp.object.integrators)
    return Deterministic(
        BMDPState(1,
            HybridBelief(
                SparseCat(1:n_discrete, fill(inv(n_discrete), n_discrete)), 
                [deepcopy(mdp.object.x) for _ in 1:n_discrete]
            )
        )
    )
end
