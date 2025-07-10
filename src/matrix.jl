# NOTE: IMPORTANT - TensorGames.jl reasons about COST matrices NOT Utility matrices

# FIXME: overloading [MarkovGames/POMDPs].solve on an AbstractMatrix seems like a bad idea
function MarkovGames.solve(A::AbstractMatrix)
    eq = compute_equilibrium([-A, A])
    x,y = normalize!(eq.x[1],1), normalize!(eq.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end

function MarkovGames.solve(A::AbstractMatrix, B::AbstractMatrix)
    eq1 = compute_equilibrium([-A, A])
    eq2 = compute_equilibrium([B, -B])
    x,y = normalize!(eq1.x[1],1), normalize!(eq2.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end

struct PATHSolver{P<:Base.Pairs}
    kwargs::P
    PATHSolver(;kwargs...) = new{typeof(kwargs)}(kwargs)
end

function MarkovGames.solve(sol::PATHSolver, A::AbstractMatrix)
    eq = compute_equilibrium([-A, A]; sol.kwargs...)
    x,y = normalize!(eq.x[1],1), normalize!(eq.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end

function MarkovGames.solve(sol::PATHSolver, A::AbstractMatrix, B::AbstractMatrix)
    eq1 = compute_equilibrium([-A, A]; sol.kwargs...)
    eq2 = compute_equilibrium([B, -B]; sol.kwargs...)
    x,y = normalize!(eq1.x[1],1), normalize!(eq2.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end

##

struct RegretSolver
    n::Int
end

function match!(x, r)
    rs = zero(eltype(r))
    for i ∈ eachindex(x,r)
        r_i = r[i]
        if r_i > zero(r_i)
            rs += r_i
            x[i] = r_i
        else
            x[i] = zero(r_i)
        end
    end
    rs > zero(rs) ? x ./= rs : x .= inv(length(x))
end

# FIXME: This assumes reward structure dot(y,A,x), whereas existing code assumes dot(x,A,y)
function regret_match(A::AbstractMatrix, x=uniform(size(A, 2)), y=uniform(size(A, 1)); n = 10)
    rx = zero(x)
    πx = copy(x)
    πy = copy(y)
    ry = zero(y)

    rxi = zero(x)
    ryi = zero(y)
    for i ∈ 1:n
        qx = mul!(rxi, A', y)
        vx = dot(qx, x)
        @. rxi = qx - vx
        x = match!(x, rx)
        
        qy = mul!(ryi, A, x)
        qy .*= -1
        vy = -vx
        @. ryi = qy - vy
        y = match!(y, ry)

        rx .+= rxi
        ry .+= ryi
        πx .+= x
        πy .+= y
    end
    x̄ = normalize!(πx, 1)
    ȳ = normalize!(πy, 1)
    v = dot(ȳ, A, x̄)
    return x̄, ȳ, v
end

MarkovGames.solve(sol::RegretSolver, A::AbstractMatrix) = regret_match(A';n=sol.n)

function MarkovGames.solve(sol::RegretSolver, A::AbstractMatrix, B::AbstractMatrix)
    xa, ya, va = regret_match(A'; n=sol.n)
    xb, yb, vb = regret_match(B'; n=sol.n)
    t = dot(xa, A, yb)
    return xa, yb, t
end
