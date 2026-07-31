const HiGHS_DEFAULTS = (;log_to_console=false)

function kwargs2attrs(kwargs)
    return map(keys(kwargs), values(kwargs)) do k,v
        string(k) => v
    end
end

kwargs2attrs(kwargs, defaults) = kwargs2attrs(merge(defaults, kwargs))

function observation_ilp_model(;n_obj=10, n_obs=2, T=5, kwargs...)
    model = Model(HiGHS.Optimizer)
    set_attributes(model, kwargs2attrs(kwargs, HiGHS_DEFAULTS)...)
    # (obj_idx, obs_idx, T_idx)
    @variable(model, X[1:n_obj, 1:n_obs, 1:T], Bin)

    # every observer can only observe at most 1 object at a time
    @constraint(model, sum(X, dims=1) .≤ 1)

    # every object must be observed at least once throughout the considered time horizon
    @constraint(model, sum(X, dims=2:3) .≥ 1)
    return model
end

function observation_ilp(;kwargs...)
    model = observation_ilp_model(;kwargs...)
    optimize!(model)
    return round.(Bool, JuMP.value.(model[:X]))
end

function observation_maximin_ilp_model(;n_obj=10, n_obs=2, T=5, kwargs...)
    model = Model(HiGHS.Optimizer)
    set_attributes(model, kwargs2attrs(kwargs, HiGHS_DEFAULTS)...)
    # (obj_idx, obs_idx, T_idx)
    @variable(model, X[1:n_obj, 1:n_obs, 1:T], Bin)
    @variable(model, min_obs_rate)
    @constraint(model, min_obs_rate .≤ sum(X, dims=2:3))
    @objective(model, Max, min_obs_rate)
    
    # every observer can only observe at most 1 object at a time
    @constraint(model, sum(X, dims=1) .≤ 1)

    return model
end

function observation_maximin_ilp(;kwargs...)
    model = observation_maximin_ilp_model(;kwargs...)
    optimize!(model)
    return round.(Bool, JuMP.value.(model[:X]))
end

"""
X[i]

Space object i has been observed a total of n times. 
"""
total_object_scans(X::AbstractArray{<:Any, 3}) = dropdims(sum(X, dims=2:3), dims=(2,3))

"""
X[i,j]

Observer i has chosen to observe n space objects at time step j. All elements should be ≤1, 
as one observer cannot choose to observe two space objects simultaneously.
"""
total_observer_scans(X::AbstractArray{<:Any, 3}) = dropdims(sum(X, dims=1), dims=1)

"""
X[i,j]

Observer i observed RSO X[i,j] on the j'th time step
"""
function condensed_observer_arr(X::BitArray{3})
    # n_obs × T
    m = zeros(Int, size(X, 2), size(X, 3))
    for i_obs ∈ axes(X,2), t ∈ axes(X,3)
        idx = findfirst(isone, X[:,i_obs,t])
        if !isnothing(idx)
            m[i_obs, t] = idx
        end
    end
    return m
end

function observation_maximin_occlusion_ilp_model(O::BitArray{3}; kwargs...)
    n_obj, n_obs, T = size(O)
    model = Model(HiGHS.Optimizer)
    set_attributes(model, kwargs2attrs(kwargs, HiGHS_DEFAULTS)...)
    # (obj_idx, obs_idx, T_idx)
    @variable(model, X[1:n_obj, 1:n_obs, 1:T], Bin)
    @variable(model, min_obs_rate)
    @constraint(model, X .≤ O)
    @constraint(model, min_obs_rate .≤ sum(X, dims=2:3))
    @objective(model, Max, min_obs_rate)
    
    # every observer can only observe at most 1 object at a time
    @constraint(model, sum(X, dims=1) .≤ 1)

    return model
end

function observation_maximin_occlusion_ilp(O; kwargs...)
    model = observation_maximin_occlusion_ilp_model(O; kwargs...)
    optimize!(model)
    return round.(Bool, JuMP.value.(model[:X]))
end
