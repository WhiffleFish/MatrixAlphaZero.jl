using Pkg
Pkg.activate("experiments")

using Distributed
using DelimitedFiles
using ExperimentTools
using MatrixAlphaZero

const AZ = MatrixAlphaZero

const STYLE_SPECS = (
    (; name = "matrix_game", label = "Greedy Matrix", style = AZ.MatrixGameSearch()),
    (; name = "regret_matching", label = "Regret Matching", style = AZ.RegretMatchingSearch()),
    (; name = "exp3", label = "Exp3", style = AZ.Exp3Search()),
)

args = ExperimentTools.parse_commandline(
    tree_queries = 1_000,
    runs = 48,
    checkpoint = 50,
    every = 50,
)

checkpoint = args["checkpoint"]
runs = args["runs"]
tree_queries = args["tree_queries"]
every = args["every"]
c = 10.0
ϵ = 0.30

p = addprocs(args["addprocs"]; exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Flux
    using POMDPTools
    using POMDPs
    using POSGModels.Dubin
    using POSGModels.StaticArrays
    using Random
end

@everywhere const SEARCH_EXPERIMENT_DIR = $(@__DIR__)

@everywhere function dubin_reference_state()
    return JointDubinState(SA[1, 1, deg2rad(45)], SA[8, 7, deg2rad(180)])
end

@everywhere function load_checkpoint_oracle(style_name::String, checkpoint::Int)
    style_dir = joinpath(SEARCH_EXPERIMENT_DIR, style_name)
    oracle = AZ.load_oracle(style_dir)
    model_path = joinpath(
        style_dir,
        "models",
        "oracle" * AZ.iter2string(checkpoint) * ".jld2",
    )
    return Flux.loadmodel!(oracle, model_path)
end

@everywhere function run_style_trial(
    style_name::String,
    style,
    checkpoint::Int,
    tree_queries::Int,
    every::Int,
    c::Float64,
    ϵ::Float64,
    trial::Int,
)
    seed = Int(hash((style_name, checkpoint, trial)) % UInt(typemax(Int)))
    Random.seed!(seed)
    game = DubinMG(V = (1.0, 1.0))
    oracle = load_checkpoint_oracle(style_name, checkpoint)
    planner = AlphaZeroPlanner(
        game,
        oracle,
        max_iter = tree_queries,
        c = c,
        search_style = style,
    )
    params = AZ.MCTSParams(planner)
    s0 = dubin_reference_state()
    result = Tools.search_eval(planner, params, game, s0; ϵ, every, progress = false)
    return (; style = style_name, trial, result...)
end

function write_style_results(base_dir::String, style_name::String, style_label::String, rows)
    style_dir = joinpath(base_dir, style_name)
    mkpath(style_dir)

    iter = rows[1].iter
    brv1 = reduce(hcat, getproperty.(rows, :brv1))
    brv2 = reduce(hcat, getproperty.(rows, :brv2))
    values = reduce(hcat, getproperty.(rows, :v))

    writedlm(joinpath(style_dir, "iter.csv"), iter, ',')
    writedlm(joinpath(style_dir, "brv1.csv"), brv1, ',')
    writedlm(joinpath(style_dir, "brv2.csv"), brv2, ',')
    writedlm(joinpath(style_dir, "value.csv"), values, ',')
    write(joinpath(style_dir, "label.txt"), style_label * "\n")
end

jobs = [
    (; style_name = spec.name, style_label = spec.label, style = spec.style, trial)
    for spec in STYLE_SPECS for trial in 1:runs
]

rows = pmap(jobs) do job
    run_style_trial(
        job.style_name,
        job.style,
        checkpoint,
        tree_queries,
        every,
        c,
        ϵ,
        job.trial,
    )
end

results_dir = joinpath(@__DIR__, "search_llbr_results")
mkpath(results_dir)

for spec in STYLE_SPECS
    style_rows = filter(row -> row.style == spec.name, rows)
    write_style_results(results_dir, spec.name, spec.label, style_rows)
end

rmprocs(p)
