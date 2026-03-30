using Pkg
Pkg.activate("experiments")

using MatrixAlphaZero
using ExperimentTools
using MarkovGames
using POSGModels.Dubin
using POSGModels.StaticArrays
using POMDPTools
using DelimitedFiles
using Random
using Statistics
using ProgressMeter

const AZ = MatrixAlphaZero
const Tools = ExperimentTools

# ---------------------------------------------------------------------------
# Zero oracle: value = 0 everywhere, policy = uniform
# Lets us isolate the search algorithm itself from any learned approximation.
# ---------------------------------------------------------------------------
struct ZeroOracle end

AZ.value(::ZeroOracle, x) = [0.0f0]
AZ.batch_state_value(::ZeroOracle, game, states) = zeros(Float64, length(states))
function AZ.state_policy(::ZeroOracle, game, s)
    na1, na2 = length.(actions(game))
    return fill(Float32(inv(na1)), na1), fill(Float32(inv(na2)), na2)
end

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
const STYLE_SPECS = (
    (; name = "matrix_game",      label = "Matrix Game",      style = AZ.MatrixGameSearch()),
    (; name = "regret_matching",  label = "Regret Matching",  style = AZ.RegretMatchingSearch()),
    (; name = "exp3",             label = "Exp3",             style = AZ.Exp3Search()),
)

tree_queries = 1_000
every        = 25       # evaluate exploitability every N simulations
max_depth    = 20       # tree depth (leaf bootstrap = 0 from ZeroOracle)
c            = 10.0
ϵ            = 0.30
runs         = 20        # independent trials per style

game   = DubinMG(V = (1.0, 1.0))
oracle = ZeroOracle()
s0     = JointDubinState(SA[1, 1, deg2rad(45)], SA[6, 6, deg2rad(-45)])

results_dir = joinpath(@__DIR__, "zero_oracle_results")
mkpath(results_dir)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for spec in STYLE_SPECS
    println("\n" * "="^60)
    println("  $(spec.label)")
    println("="^60)

    style_dir = joinpath(results_dir, spec.name)
    mkpath(joinpath(style_dir, "models"))   # unused but kept for convention

    all_brv1   = Vector{Float64}[]
    all_brv2   = Vector{Float64}[]
    all_values = Vector{Float64}[]
    iter       = Int[]

    p = Progress(runs; desc = "  trials: ", showspeed = true, color = :cyan)

    for trial in 1:runs
        Random.seed!(trial)

        planner = AlphaZeroPlanner(
            game, oracle;
            max_iter     = tree_queries,
            max_depth    = max_depth,
            c            = c,
            search_style = spec.style,
        )
        params = AZ.MCTSParams(planner)

        result = Tools.search_eval(planner, params, game, s0; ϵ, every, progress = false)

        if isempty(iter)
            iter = result.iter
        end
        push!(all_brv1,   result.brv1)
        push!(all_brv2,   result.brv2)
        push!(all_values, result.v)

        final_expl = 0.5 * (-result.brv2[end] - result.brv1[end])
        running_expl = mean(
            0.5 * (-b2[end] - b1[end])
            for (b1, b2) in zip(all_brv1, all_brv2)
        )
        next!(p; showvalues = [
            (:trial,            "$trial / $runs"),
            (:expl_this_trial,  round(final_expl;   digits = 5)),
            (:expl_running_mean, round(running_expl; digits = 5)),
        ])
    end

    brv1_mat   = reduce(hcat, all_brv1)    # (n_evals × runs)
    brv2_mat   = reduce(hcat, all_brv2)
    values_mat = reduce(hcat, all_values)

    # Exploitability = 0.5 * (v_br1 - v_br2) from P1's perspective
    # approx_br_values_both_st returns (v1_br2, -v2_br1), so:
    #   brv1 = v1_br2  (P1's value when P2 best-responds — lower bound)
    #   brv2 = -v2_br1 (P2's value when P1 best-responds — upper bound negated)
    # Nash gap = 0.5 * (-brv2 - brv1)  [≥ 0, 0 at equilibrium]
    expl_mat   = @. 0.5 * (-brv2_mat - brv1_mat)
    expl_mean  = vec(mean(expl_mat; dims = 2))

    println("\n  Queries | Exploitability (mean over $runs trials)")
    println("  --------+-" * "-"^30)
    for (n, e) in zip(iter, expl_mean)
        println("  $(lpad(n, 7)) | $(round(e, digits=5))")
    end

    writedlm(joinpath(style_dir, "iter.csv"),   iter,        ',')
    writedlm(joinpath(style_dir, "brv1.csv"),   brv1_mat,    ',')
    writedlm(joinpath(style_dir, "brv2.csv"),   brv2_mat,    ',')
    writedlm(joinpath(style_dir, "value.csv"),  values_mat,  ',')
    write(joinpath(style_dir, "label.txt"),     spec.label * "\n")
end

println("\n\nResults saved to: $(results_dir)")
