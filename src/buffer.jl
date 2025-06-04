struct Buffer
    s::CircularBuffer{Vector{Float32}}
    v::CircularBuffer{Float32}
    p::NTuple{2,CircularBuffer{Vector{Float32}}}
end

Buffer(l::Int) = Buffer(
    CircularBuffer{Vector{Float32}}(l), 
    CircularBuffer{Float32}(l),
    (CircularBuffer{Vector{Float32}}(l), CircularBuffer{Vector{Float32}}(l))
)

function Base.push!(buf::Buffer, t::NamedTuple)
    @assert length(t.v) == length(t.s) == length(t.policy[1]) == length(t.policy[2])
    append!(buf.s, t.s)
    append!(buf.v, t.v)
    append!(buf.p[1], t.policy[1])
    append!(buf.p[2], t.policy[2])
    return buf
end

Base.length(buf::Buffer) = length(buf.v)

Base.eachindex(buf::Buffer) = Base.OneTo(length(buf))

function Base.getindex(buf::Buffer, args...)
    return (
        s = buf.s[args...],
        v = buf.v[args...],
        p = map(buf.p) do p 
            p[args...]
        end
    )
end
