@proto struct Buffer
    const s::CircularBuffer{Vector{Float32}}
    const v::CircularBuffer{Float32}
end

Buffer(l::Int) = Buffer(CircularBuffer{Vector{Float32}}(l), CircularBuffer{Float32}(l))

function Base.push!(buf::Buffer, t::NamedTuple)
    @assert length(t.v) == length(t.s)
    append!(buf.s, t.s)
    append!(buf.v, t.v)
    return buf
end

Base.length(buf::Buffer) = length(buf.v)
