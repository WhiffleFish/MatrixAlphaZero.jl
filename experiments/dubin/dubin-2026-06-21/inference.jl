using Pkg
Pkg.activate("experiments")

using ExperimentTools
using Flux
using MarkovGames
using MatrixAlphaZero
using Plots
using POMDPTools
using POMDPs
using POSGModels.Dubin
using POSGModels.StaticArrays
using Random

const AZ = MatrixAlphaZero

const SEARCH_NAME = get(ENV, "DUBIN_SEARCH", "smoos")
const MODEL_ITER = get(ENV, "DUBIN_ITER", "latest")
const TREE_QUERIES = parse(Int, get(ENV, "DUBIN_TREE_QUERIES", "250"))
const TRAIN_OOS_ITERATIONS = parse(Int, get(ENV, "DUBIN_TRAIN_OOS_ITERATIONS", "1000"))
const SEARCH_DEPTH = parse(Int, get(ENV, "DUBIN_SEARCH_DEPTH", "5"))
const MAX_STEPS = parse(Int, get(ENV, "DUBIN_MAX_STEPS", "50"))
const ROLLOUTS = parse(Int, get(ENV, "DUBIN_ROLLOUTS", "10"))
const GIF_FPS = parse(Int, get(ENV, "DUBIN_GIF_FPS", "2"))
const TRAJECTORY_POINTS_PER_STEP = parse(Int, get(ENV, "DUBIN_TRAJECTORY_POINTS_PER_STEP", "12"))
const RNG_SEED = parse(Int, get(ENV, "DUBIN_SEED", "0"))

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

experiment_dir = @__DIR__
figures_dir = joinpath(experiment_dir, "figures")
mkpath(figures_dir)

function checkpoint_iteration(path::AbstractString)
    m = match(r"oracle(\d+)\.jld2$", basename(path))
    isnothing(m) && error("Cannot parse checkpoint iteration from $(path)")
    return parse(Int, m.captures[1])
end

function model_paths(search_name::AbstractString)
    oracle_file = joinpath(experiment_dir, "oracle_$(search_name).jld2")
    models_dir = joinpath(experiment_dir, "models_$(search_name)")
    isfile(oracle_file) || error("Missing oracle architecture file: $(oracle_file)")
    isdir(models_dir) || error("Missing model checkpoint directory: $(models_dir)")
    checkpoints = filter(p -> occursin(r"oracle\d+\.jld2$", basename(p)), readdir(models_dir; join=true))
    isempty(checkpoints) && error("No oracle checkpoints found in $(models_dir)")
    sort!(checkpoints; by=checkpoint_iteration)
    return oracle_file, checkpoints
end

function select_checkpoint(checkpoints::Vector{String}, iter_spec::AbstractString)
    if iter_spec == "latest"
        return last(checkpoints)
    end
    iter = parse(Int, iter_spec)
    matches = filter(p -> checkpoint_iteration(p) == iter, checkpoints)
    isempty(matches) && error("No checkpoint for iteration $(iter)")
    return only(matches)
end

function load_checkpoint_oracle(search_name::AbstractString, iter_spec::AbstractString)
    oracle_file, checkpoints = model_paths(search_name)
    checkpoint = select_checkpoint(checkpoints, iter_spec)
    oracle = AZ.load_oracle(oracle_file)
    Flux.loadmodel!(oracle, checkpoint)
    return oracle, checkpoint_iteration(checkpoint), checkpoint
end

function smoos_transfer_tau(iter::Integer; oos_iterations::Integer, transfer_weight::Real)
    tau = 0.0
    for _ in 1:iter
        tau = AZ.advance_transfer_tau(tau, oos_iterations, transfer_weight)
    end
    return tau
end

function make_search(search_name::AbstractString, oracle, iter::Integer)
    epsilon = _ -> 0.0
    if search_name == "smoos"
        transfer_weight = 0.1
        tau = smoos_transfer_tau(iter; oos_iterations=TRAIN_OOS_ITERATIONS, transfer_weight)
        return AZ.SMOOSSearch(;
            oracle,
            oos_iterations = TREE_QUERIES,
            max_depth = SEARCH_DEPTH,
            transfer_weight,
            τ = tau,
            ϵ = epsilon,
        )
    elseif search_name == "regret_matching"
        return AZ.MCTSSearch(;
            oracle,
            tree_queries = TREE_QUERIES,
            max_depth = SEARCH_DEPTH,
            max_time = Inf,
            search_style = AZ.RegretMatchingSearch(; backup=:sample),
            value_target = :search,
            ϵ = epsilon,
        )
    else
        error("Unsupported DUBIN_SEARCH=$(search_name). Use smoos or regret_matching.")
    end
end

function search_summary(search::AZ.SMOOSSearch)
    return "SMOOSSearch(oos_iterations=$(search.oos_iterations), max_depth=$(search.max_depth), transfer_weight=$(search.transfer_weight), tau=$(round(search.τ; digits=3)))"
end

function search_summary(search::AZ.MCTSSearch)
    return "MCTSSearch(tree_queries=$(search.tree_queries), max_depth=$(search.max_depth), backup=$(search.search_style.backup), value_target=$(search.value_target))"
end

function initial_dubin_state()
    return JointDubinState(
        SA[1, 1, deg2rad(45)],
        SA[8, 7, deg2rad(180)],
    )
end

function run_rollouts(game, planner, s0; n::Integer, max_steps::Integer)
    simulator = HistoryRecorder(max_steps=max_steps)
    return [simulate(simulator, game, planner, s0) for _ in 1:n]
end

state_at(step) = haskey(step, :s) ? step[:s] : step.s
behavior_at(step) = haskey(step, :behavior) ? step[:behavior] : step.behavior
action_at(step) = haskey(step, :a) ? step[:a] : step.a

function save_rollout_gif(game, history, path::AbstractString)
    anim = @animate for step in history
        plot(
            game,
            state_at(step),
            behavior_at(step);
            title = "Dubin $(SEARCH_NAME) iter $(model_iter)",
            aspect_ratio = 1.0,
            size = (600, 600),
        )
    end
    gif(anim, path; fps=GIF_FPS)
    return path
end

function dubin_segment_coords(game, x, action_idx::Integer, player::Integer; n::Integer)
    turn_rate = game.actions[player][action_idx]
    speed = game.V[player]
    pts = map(range(0, game.dt; length=n)) do dt
        sp = Dubin.force_inbounds(Dubin.dubinstep(x, turn_rate, speed, dt), game.floor)
        return sp[1], sp[2]
    end
    return first.(pts), last.(pts)
end

function trajectory_coords(game, history, player::Symbol; n::Integer=TRAJECTORY_POINTS_PER_STEP)
    player_idx = player == :attacker ? 1 : 2
    xs = Float64[]
    ys = Float64[]
    for step in history
        s = state_at(step)
        x = getfield(s, player)
        seg_x, seg_y = dubin_segment_coords(game, x, action_at(step)[player_idx], player_idx; n)
        append!(xs, seg_x)
        append!(ys, seg_y)
    end
    return xs, ys
end

function save_trajectory_plot(game, histories, path::AbstractString)
    plt = plot(
        game.goal;
        xlims = (0, game.floor[1] + 1),
        ylims = (0, game.floor[2] + 1),
        aspect_ratio = 1.0,
        size = (650, 650),
        title = "Dubin $(SEARCH_NAME) iter $(model_iter): $(length(histories)) rollouts",
    )
    for (i, hist) in enumerate(histories)
        ax, ay = trajectory_coords(game, hist, :attacker)
        dx, dy = trajectory_coords(game, hist, :defender)
        plot!(plt, ax, ay; c=:blue, alpha=0.35, lw=2, label=i == 1 ? "attacker" : "")
        plot!(plt, dx, dy; c=:red, alpha=0.35, lw=2, label=i == 1 ? "defender" : "")
        scatter!(plt, [first(ax)], [first(ay)]; c=:blue, ms=4, label="")
        scatter!(plt, [first(dx)], [first(dy)]; c=:red, ms=4, label="")
    end
    savefig(plt, path)
    return path
end

Random.seed!(RNG_SEED)

game = DubinMG(V=(1.0, 1.0))
s0 = initial_dubin_state()
oracle, model_iter, checkpoint = load_checkpoint_oracle(SEARCH_NAME, MODEL_ITER)
search = make_search(SEARCH_NAME, oracle, model_iter)
planner = AZ.AlphaZeroPlanner(game, search)
histories = run_rollouts(game, planner, s0; n=ROLLOUTS, max_steps=MAX_STEPS)

stem = "trajectory_$(SEARCH_NAME)_oracle$(AZ.iter2string(model_iter))_q$(TREE_QUERIES)_d$(SEARCH_DEPTH)_n$(ROLLOUTS)_h$(MAX_STEPS)"
gif_path = joinpath(figures_dir, stem * ".gif")
png_path = joinpath(figures_dir, stem * "_rollouts.png")
pdf_path = joinpath(figures_dir, stem * "_rollouts.pdf")

save_rollout_gif(game, first(histories), gif_path)
save_trajectory_plot(game, histories, png_path)
save_trajectory_plot(game, histories, pdf_path)

println("Loaded checkpoint: $(checkpoint)")
println("Search: $(search_summary(search))")
println("Rollouts: $(length(histories)) x $(MAX_STEPS) max steps")
println("GIF: $(gif_path)")
println("Trajectory plot: $(png_path)")
