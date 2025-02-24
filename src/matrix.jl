function solve(A::AbstractMatrix)
    eq = compute_equilibrium([A, -A])
    x,y = normalize!(eq.x[1],1), normalize!(eq.x[2],1)
    t = dot(x, A, y)
    return x,y,t
end
