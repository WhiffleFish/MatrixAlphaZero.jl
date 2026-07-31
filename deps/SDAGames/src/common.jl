const idx1t3 = StaticArrays.SUnitRange(1,3)
const idx4t6 = StaticArrays.SUnitRange(4,6)

const FLOAT_RANGE = StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}

const StateVec = SVector{6, Float64}

const DEFAULT_INTEGRATOR_OPTIONS = (;
    n_grav      = 0,
    m_grav      = 0,
    drag        = false,
    srp         = false,
    moon        = false,
    sun         = false,
    relativity  = false,
    mass        = 1.0, 
    area_drag   = 1.0, 
    coef_drag   = 2.3, 
    area_srp    = 1.0, 
    coef_srp    = 1.8
)

struct Observer{NT<:NamedTuple, XT<:AbstractVector}
    x::XT
    ground_based::Bool
    rk4::RK4
    function Observer(x; ground_based=false, kwargs...)
        opts = merge(DEFAULT_INTEGRATOR_OPTIONS, kwargs)
        return new{typeof(opts), typeof(x)}(x, ground_based, RK4(SatelliteDynamics.fderiv_earth_orbit, merge(DEFAULT_INTEGRATOR_OPTIONS, opts)))
    end
end

function apply_dv(x::AbstractVector, Δv)
    Δvp = first(Δv)
    v = x[idx4t6]
    v̂ = normalize(v, 2)
    return vcat(x[idx1t3], v .+ Δvp .* v̂)
end
