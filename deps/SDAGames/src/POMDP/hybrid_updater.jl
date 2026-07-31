# struct HybridUpdater{GU<:AbstractVector{<:LLPF.AbstractFilter}} <: Updater 
#     GaussianUp::GU  # vector of KF type filters, e.g. UKF using LowLevelParticleFilters
# end

# struct JointUpdater{HU<:HybridUpdater , GU<:AbstractVector{<:LLPF.AbstractFilter}} <: Updater
#     HybridUp::HU
#     CatalogUp::GU
# end

struct HybridState{V<:AbstractVector}
    discrete::Int
    continuous::V
end

function Base.rand(rng::AbstractRNG, s::Random.SamplerTrivial{<:HybridBelief})
    b = s[]
    d_idx = rand(rng, b.discrete)
    return HybridState(d_idx, rand(rng, b.continuous[d_idx]))
end


function gen_state(mdp::SDABMDP, s::BMDPState, a::Int, rng=Random.default_rng())
    bp = predict(mdp, s, a)
    _s = rand(rng, bp)
    t = s.t + 1
    x_obj = _s.continuous
    if iszero(a)
        return BMDPState(t, bp)
    else
        x_obs = mdp.observers[a][:, t]
        o = rand(rng, MvNormal(_measurement(x_obj, x_obs), mdp.obs_noise))
        return BMDPState(t, correct(mdp, a, BMDPState(t, bp), o))
    end
end
# struct JointBelief{V<:AbstractArray}
#     v::V
# end

# function POMDPs.update(up::LLPF.AbstractFilter, b::HybridBelief, a, o)
#     b_d_p = similar(b.discrete)
#     b_c_p = similar(b.continuous)
#     for i ∈ eachindex(b.discrete, b.continuous)
#         b_d_i, b_c_i = b.discrete[i], b.continuous[i]
#         b_c_p, info = update(up.GaussianUp[i], b_c_i, a, o)
#         b_d_p[i] = b_d_i[i] * exp(info.ll)
#     end
#     b_d_p ./= sum(b_d_p)
#     return HybridBelief(b_d_p, b_c_p)
# end 

# POMDPs.initialize_belief(::SDAPOMDP, b::JointBelief) = b



function predict(mdp::SDABMDP, s, a)
    (;t, b) = s

    
    b_d_p = copy(b.discrete)
    b_c_p = similar(b.continuous)
    for i ∈ eachindex(b.continuous)
        p = (; rk4=mdp.object.integrators[i], epc=mdp.epcs[t], dt=step(mdp.ts))
        b_c_p[i] = predict(mdp.kf, b.continuous[i], a, p)
    end
    return HybridBelief(b_d_p, b_c_p)
end

function correct(mdp::SDABMDP, a, sp, o)
    iszero(a) && return b
    (;t, b) = sp
    p = (; x_obs=mdp.observers[a][:, t])
    
    b_d_p = copy(b.discrete)
    b_c_p = similar(b.continuous)
    for i ∈ eachindex(b.continuous)
        b_c_p_i, info = correct(mdp.kf, b.continuous[i], a, o, p)
        b_c_p[i] = b_c_p_i
        b_d_p.probs[i] = b_d_p.probs[i] * info.l
    end
    b_d_p.probs ./= sum(b_d_p.probs)
    return HybridBelief(b_d_p, b_c_p)
end

function LLPF.sigmapoints!(xs, m, Σ::PDMat)
    n = length(xs[1])
    @assert n == length(m)
    X = sqrt(Symmetric(n*Σ)) # 2.184 μs (16 allocations: 2.27 KiB)
    # X = cholesky!(Symmetric(n*Σ)).L # 170.869 ns (3 allocations: 176 bytes)
    @inbounds @views for i in 1:n
        xs[i] = X[:,i]
        xs[i+n] = -xs[i] .+ m
        xs[i] = xs[i] .+ m
    end
    xs[end] = m
    xs
end


function POMDPs.update(mdp::SDABMDP, s::BMDPState, a::Int, o)
    t = s.t + 1
    bp = predict(mdp, s, a)

    if any(isnan,o)
        return BMDPState(t, bp)
    else
        bp_p = correct(mdp, a, BMDPState(t, bp), o)   
        return BMDPState(t, bp_p)
    end
end 
