# TODO: should probably be called BestResponseData
struct ExploitabilityData
    br::NTuple{2,Matrix{Float64}}
end

Base.getindex(ed::ExploitabilityData, args...) = getindex(ed.br, args...)
Base.length(ed::ExploitabilityData) = length(ed.br)

function ExploitabilityData(dir::String)
    return ExploitabilityData((
        DelimitedFiles.readdlm(joinpath(dir, "br1.csv"), ','),
        DelimitedFiles.readdlm(joinpath(dir, "br2.csv"), ',')
    ))
end

function Statistics.mean(ed::ExploitabilityData)
    return map(ed.br) do br_i
        vec(mean(br_i, dims=2))
    end
end

function Statistics.std(ed::ExploitabilityData)
    return map(ed.br) do br_i
        vec(std(br_i, dims=2))
    end
end

n_sims(ed::ExploitabilityData) = size(first(ed.br), 2)

function stderror(ed::ExploitabilityData)
    return map(ed.br) do br_i
        n = size(br_i, 2)
        vec(std(br_i, dims=2)) ./ √n
    end
end

@recipe function f(ed::ExploitabilityData)
    means = mean(ed)
    stds = std(ed)
    layout --> (2,1)
    
    for (i,(μ_i, σ_i)) ∈ enumerate(zip(means, stds))
        xs = eachindex(μ_i) .- 1
        @series begin
            subplot := i
            seriestype := :path
            primary := false
            linecolor := nothing
            fillcolor := :lightgray
            fillalpha := 0.5
            fillrange := μ_i .- σ_i
            # ensure no markers are shown for the error band
            markershape := :none
            # return series data
            xs, μ_i .+ σ_i
        end
        @series begin
            subplot := i
            seriestype := :path
            xlabel --> "Training Iteration"
            ylabel --> "BRV"
            label --> nothing
            xs, μ_i
        end
    end
end

struct NashConvData
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function NashConvData(ed::ExploitabilityData)
    return NashConvData(
        reduce(+, mean(ed)),
        mapreduce(+, std(ed)) do σ_i
            σ_i .^ 2
        end .|> sqrt
    )
end

@recipe function f(nc::NashConvData)
    xs = eachindex(nc.μ) .- 1
    label --> nothing
    xlabel --> "Training Iteration"
    ylabel --> "NashConv"

    @series begin
        seriestype := :path
        primary := false
        linecolor := nothing
        fillcolor := :lightgray
        fillalpha := 0.5
        fillrange := nc.μ .- nc.σ
        # ensure no markers are shown for the error band
        markershape := :none
        # return series data
        xs, nc.μ .+ nc.σ
    end

    xs, nc.μ
end
