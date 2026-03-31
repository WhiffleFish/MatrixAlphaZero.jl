using Pkg
Pkg.activate("experiments")

using Distributed
using DelimitedFiles
using ExperimentTools
using MatrixAlphaZero
using ProgressMeter

const AZ = MatrixAlphaZero

const STYLE_SPECS = (
    (; name = "matrix_game", label = "Greedy Matrix", style = AZ.MatrixGameSearch()),
    (; name = "regret_matching", label = "Regret Matching", style = AZ.RegretMatchingSearch()),
    (; name = "exp3", label = "Exp3", style = AZ.Exp3Search()),
)

args = ExperimentTools.parse_commandline(
    tree_queries = 1_000,
    runs = 48,
    checkpoint = 0,
    every = 50,
)

function available_checkpoints(style_name::String)
    model_dir = joinpath(@__DIR__, style_name, "models")
    if !isdir(model_dir)
        return Int[]
    end
    checkpoints = Int[]
    for filename in readdir(model_dir)
        m = match(r"^oracle(\d+)\.jld2$", filename)
        if !isnothing(m)
            push!(checkpoints, parse(Int, only(m.captures)))
        end
    end
    sort!(checkpoints)
    return checkpoints
end

function resolve_checkpoint(requested_checkpoint::Int)
    style_names = collect(getproperty.(STYLE_SPECS, :name))
    checkpoint_sets = Dict(name => Set(available_checkpoints(name)) for name in style_names)
    if iszero(requested_checkpoint)
        common = copy(checkpoint_sets[first(style_names)])
        for name in style_names[2:end]
            intersect!(common, checkpoint_sets[name])
        end
        if isempty(common)
            details = join(
                ["$(name): $(collect(checkpoint_sets[name]))" for name in style_names],
                ", ",
            )
            error("No common checkpoint found across styles. Available checkpoints: $(details)")
        end
        return maximum(common)
    end

    missing = [name for name in style_names if requested_checkpoint ∉ checkpoint_sets[name]]
    if !isempty(missing)
        details = join(
            ["$(name): $(collect(checkpoint_sets[name]))" for name in missing],
            ", ",
        )
        error("Checkpoint $(requested_checkpoint) is unavailable for $(join(missing, ", ")). Available checkpoints: $(details)")
    end
    return requested_checkpoint
end

checkpoint = resolve_checkpoint(args["checkpoint"])
runs = args["runs"]
tree_queries = args["tree_queries"]
every = args["every"]
c = 10.0
ϵ = 0.30
progress_batch = min(10, max(tree_queries, 1))

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

@everywhere function run_style_trial_with_progress(
    style_name::String,
    style,
    checkpoint::Int,
    tree_queries::Int,
    every::Int,
    c::Float64,
    ϵ::Float64,
    trial::Int,
    progress_channel,
    progress_batch::Int,
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
    tree = AZ.Tree(params, game, s0)
    iter = Int[]
    brvs1 = Float64[]
    brvs2 = Float64[]
    values = Float64[]
    pending_progress = 0

    try
        for i in 1:tree_queries
            AZ.simulate(params, tree, game, 1; ϵ)
            pending_progress += 1
            if pending_progress >= progress_batch
                put!(progress_channel, pending_progress)
                pending_progress = 0
            end
            if iszero(mod(i, every))
                π1 = Tools.policy1_from_tree(game, planner, tree)
                π2 = Tools.policy2_from_tree(game, planner, tree)
                brv1, brv2 = Tools.approx_br_values_both_st(game, planner.oracle, π1, π2, s0; max_depth = 5)
                x, y = AZ.tree_policy(params, tree, game, 1)
                t = AZ.node_value(params, tree, game, 1, x, y)
                push!(iter, i)
                push!(brvs1, brv1)
                push!(brvs2, brv2)
                push!(values, t)
            end
        end
    finally
        if !iszero(pending_progress)
            put!(progress_channel, pending_progress)
        end
    end

    return (;
        style = style_name,
        trial,
        iter,
        brv1 = brvs1,
        brv2 = brvs2,
        v = values,
    )
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

total_progress = length(jobs) * tree_queries
progress = Progress(total_progress; desc = "LLBR search", showspeed = true)
progress_channel = RemoteChannel(() -> Channel{Int}(1024), 1)
progress_task = @async begin
    completed = 0
    while true
        delta = take!(progress_channel)
        if delta < 0
            break
        end
        completed += delta
        update!(progress, completed)
    end
end

println("Using checkpoint $(checkpoint) across $(length(STYLE_SPECS)) search styles, $(runs) runs each, $(tree_queries) queries per run.")

rows = try
    pmap(jobs) do job
        run_style_trial_with_progress(
            job.style_name,
            job.style,
            checkpoint,
            tree_queries,
            every,
            c,
            ϵ,
            job.trial,
            progress_channel,
            progress_batch,
        )
    end
finally
    put!(progress_channel, -1)
    wait(progress_task)
    finish!(progress)
end

results_dir = joinpath(@__DIR__, "search_llbr_results")
mkpath(results_dir)

for spec in STYLE_SPECS
    style_rows = filter(row -> row.style == spec.name, rows)
    write_style_results(results_dir, spec.name, spec.label, style_rows)
end

rmprocs(p)
