using Pkg
Pkg.activate("experiments")

using Distributed
using DelimitedFiles
using ExperimentTools
using MatrixAlphaZero
using ProgressMeter

const AZ = MatrixAlphaZero
const SEARCH_NAME = "regret_matching"
const SEARCH_LABEL = "Regret Matching"

args = ExperimentTools.parse_commandline(
    tree_queries = 1_000,
    runs = 48,
    checkpoint = 0,
    every = 50,
)

function available_checkpoints()
    model_dir = joinpath(@__DIR__, SEARCH_NAME, "models")
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
    checkpoints = available_checkpoints()
    if iszero(requested_checkpoint)
        if isempty(checkpoints)
            error("No checkpoints found in $(joinpath(@__DIR__, SEARCH_NAME, "models")).")
        end
        return maximum(checkpoints)
    end

    if requested_checkpoint ∉ checkpoints
        error("Checkpoint $(requested_checkpoint) is unavailable. Available checkpoints: $(checkpoints)")
    end
    return requested_checkpoint
end

checkpoint = resolve_checkpoint(args["checkpoint"])
runs = args["runs"]
tree_queries = args["tree_queries"]
every = args["every"]
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

@everywhere const SEARCH_NAME = $SEARCH_NAME

@everywhere function load_checkpoint_oracle(checkpoint::Int)
    style_dir = joinpath(SEARCH_EXPERIMENT_DIR, SEARCH_NAME)
    oracle = AZ.load_oracle(style_dir)
    model_path = joinpath(
        style_dir,
        "models",
        "oracle" * AZ.iter2string(checkpoint) * ".jld2",
    )
    return Flux.loadmodel!(oracle, model_path)
end

@everywhere function run_style_trial_with_progress(
    checkpoint::Int,
    tree_queries::Int,
    every::Int,
    ϵ::Float64,
    trial::Int,
    progress_channel,
    progress_batch::Int,
)
    seed = Int(hash((SEARCH_NAME, checkpoint, trial)) % UInt(typemax(Int)))
    Random.seed!(seed)
    game = DubinMG(V = (1.0, 1.0))
    oracle = load_checkpoint_oracle(checkpoint)
    planner = AlphaZeroPlanner(
        game,
        oracle,
        max_iter = tree_queries,
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
        style = SEARCH_NAME,
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

jobs = [(; trial) for trial in 1:runs]

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

println("Using checkpoint $(checkpoint) for $(SEARCH_LABEL), $(runs) runs, $(tree_queries) queries per run.")

rows = try
    pmap(jobs) do job
        run_style_trial_with_progress(
            checkpoint,
            tree_queries,
            every,
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

write_style_results(results_dir, SEARCH_NAME, SEARCH_LABEL, rows)

rmprocs(p)
