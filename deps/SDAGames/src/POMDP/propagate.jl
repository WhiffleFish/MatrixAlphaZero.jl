function propagate_full(orb::EarthInertialState, Tf)
    t, epc, eci = SatelliteDynamics.sim(orb, Tf)
    return eci
end

function propagate_full(orbs::AbstractArray{<:EarthInertialState}, Tf)
    return map(orbs) do orb
        propagate_full(orb, Tf)
    end
end

SatelliteDynamics.istep(orb::EarthInertialState, epc, x) = istep(orb.rk4, epc, orb.dt, x)

function Base.step(pomdp::SDAPOMDP, X::Matrix{Float64}, t_idx::Int)
    epc = pomdp.epcs[t_idx]
    dt = step(pomdp)
    return mapreduce(hcat, eachcol(X)) do x_i
        istep(pomdp.rk4, epc, dt, x_i)
    end
end

function POMDPs.gen(pomdp::SDAPOMDP, s::State, a, rng=Random.default_rng())
    return (;
        sp = State(step(pomdp, s.X, s.t), s.t+1)
    )
end

## ground-based
function Txyz2XYZ(t, α0=0.0)
    α = α0 + ω_e*t
    return SA[
        cos(α)  -sin(α)  0;
        sin(α)   cos(α)  0;
        0        0       1
    ]
end

function Txyz2XYZ_ang(α)
    return SA[
        cos(α)  -sin(α)  0;
        sin(α)   cos(α)  0;
        0        0       1
    ]
end

xyz2XYZ(r, t) = Txyz2XYZ(t) * r
xyz2XYZ_ang(r, θ) = Txyz2XYZ_ang(θ) * r

ω_e = SA[0,0,2π/(24 * 60 * 60)]

function propagate_full_ground(orb::EarthInertialState, Tf)
    ts = 0:orb.dt:Tf
    R0 = orb.x[1:3]
    θs = map(ts) do t
        ω_e .* t
    end
    return mapreduce(Array ∘ hcat, θs) do θ
        r = xyz2XYZ_ang(R0, last(θ))
        v = ω_e × r
        [r;v]
    end
end
