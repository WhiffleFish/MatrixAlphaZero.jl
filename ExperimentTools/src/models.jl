struct ModelLibrary{P}
    planner::P
    modelpaths::Vector{String}
    function ModelLibrary(planner::P, dir) where P<:AlphaZeroPlanner
        return new{P}(planner,readdir(dir; join=true))
    end
end

Base.length(ml::ModelLibrary) = length(ml.modelpaths)
Base.firstindex(ml::ModelLibrary) = 0
Base.lastindex(ml::ModelLibrary) = length(ml) - 1
Base.eachindex(ml::ModelLibrary) = firstindex(ml):lastindex(ml)

function Base.checkbounds(ml::ModelLibrary, i::Int)
    checkindex(Bool, eachindex(ml), i) || error("Bounds Error: $i ∉ $(eachindex(ml))")
    nothing
end

function Base.getindex(ml::ModelLibrary, i::Int)
    @boundscheck checkbounds(ml, i)
    modelpath = ml.modelpaths[i+1]
    planner = deepcopy(ml.planner)
    Flux.loadmodel!(deepcopy(planner), modelpath)
    return planner
end
