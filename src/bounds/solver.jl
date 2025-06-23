@kwdef struct BoundSolver{VL, VU} <: Solver
    max_iter        :: Int      = 10
    ϵ               :: Float64  = 0.80
    max_depth       :: Int      = 50
    v_lower::VL
    v_upper::VU
end
