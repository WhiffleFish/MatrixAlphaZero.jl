function solve(A::AbstractMatrix)
    eq = compute_equilibrium([A, -A])
    return eq.x[1], eq.x[2]
end
