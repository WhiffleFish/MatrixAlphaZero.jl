# NOTE: IMPORTANT - TensorGames.jl reasons about COST matrices NOT Utility matrices

function solve(A::AbstractMatrix)
    eq = compute_equilibrium([-A, A])
    x,y = normalize!(eq.x[1],1), normalize!(eq.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end

function solve(A::AbstractMatrix, B::AbstractMatrix)
    eq1 = compute_equilibrium([-A, A])
    eq2 = compute_equilibrium([B, -B])
    x,y = normalize!(eq1.x[1],1), normalize!(eq2.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end
