function to_xy_scatter(tup_arr, idx)
    [map(arr->arr[1,idx], tup_arr)...], [map(arr->arr[2,idx], tup_arr)...]
end

function sightlines(X_obs, X_obj, O::BitArray{3}, t)
    n_obs = length(X_obs)
    n_obj = length(X_obj)
    Xs = Vector{Float64}[]
    Ys = Vector{Float64}[]
    
    for i ∈ 1:n_obj, j ∈ 1:n_obs
        if O[i,j,t]
            push!(Xs, [X_obs[j][1,t], X_obj[i][1,t]])
            push!(Ys, [X_obs[j][2,t], X_obj[i][2,t]])
        end
    end
    return mapreduce(Array ∘ transpose, vcat, Xs)', mapreduce(Array ∘ transpose, vcat, Ys)'
end

function sightlines(X_obs, X_obj, O::Matrix{Int}, t)
    Xs = Vector{Float64}[]
    Ys = Vector{Float64}[]

    for (i,j) in enumerate(@view(O[t,:]))
        if !iszero(j)
            push!(Xs, [X_obs[i][1,t], X_obj[j][1,t]])
            push!(Ys, [X_obs[i][2,t], X_obj[j][2,t]])
        end
    end
    if !isempty(Xs)
        Xs = mapreduce(Array ∘ transpose, vcat, Xs)'
    end
    if !isempty(Ys)
        Ys = mapreduce(Array ∘ transpose, vcat, Ys)'
    end
    return Xs, Ys
end

struct SDAFrames{T<:AbstractArray}
    X_obs
    X_obj
    Xilp::T
end

const SDABitFrames = SDAFrames{BitArray{3}}
const SDACondensedFrames = SDAFrames{Matrix{Int}}

Base.length(frames::SDABitFrames) = size(frames.Xilp, 3)
Base.length(frames::SDACondensedFrames) = size(frames.Xilp, 1)

circle(n = 20, r = 1) =
    [(r * cos(u), r * sin(u)) for u in range(0, stop = 2π, length = n)]

@recipe function f(frames::SDAFrames, t::Int; sqlim=nothing, sqsize=nothing, obj_marker_color=:red)
    grid        --> false
    framestyle  --> :box
    fontfamily  --> "Computer Modern"
    labels      --> ""
    

    (;X_obs, X_obj, Xilp) = frames
    _X_obs = to_xy_scatter(X_obs, t)
    _X_obj = to_xy_scatter(X_obj, t)
    if !isnothing(sqlim)
        xlims := (-sqlim, sqlim)
        ylims := (-sqlim, sqlim)
        aspect_ratio --> 1
    end
    if !isnothing(sqsize)
        size --> (sqsize, sqsize)
        markersize --> sqsize / 100
        linewidth --> sqsize / 100
        legendfontsize --> sqsize / 50
    end
    
    @series begin
        seriestype := :shape
        fillcolor := :turquoise
        linewidth := 0
        circle(100, R_EARTH)
    end
    @series begin
        c := get(plotattributes, :linecolor, :blue)
        lw := get(plotattributes, :linewidth, 5)
        alpha := 1.0
        sightlines(X_obs, X_obj, Xilp, t)
    end
    @series begin
        seriestype := :path
        linecolor := nothing
        markercolor := :blue
        markershape --> :diamond
        label := "Observers"
        _X_obs[1], _X_obs[2]
    end
    @series begin
        seriestype := :path
        linecolor := nothing
        markercolor --> obj_marker_color
        markershape --> :diamond
        label := "RSO"
        _X_obj[1], _X_obj[2]
    end
    
end
