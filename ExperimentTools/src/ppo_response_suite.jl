# Approximate best responses against fixed search planners.
#
# For a zero-sum game, NashConv(σ) = Σᵢ [maxᵣ uᵢ(r, σ₋ᵢ) - uᵢ(σ)], so the value of any
# fixed responder r against σ₋ᵢ lower-bounds the best-response value. Each seat trains
# PPO responders, evaluates them alongside other fixed responders (the solvers' own seats
# and domain heuristics) on held-out selection rollouts, and reports the selected
# responder on fresh evaluation rollouts. The solver's own seat is always a candidate, so
# the reported per-seat gain over σᵢ is nonnegative up to evaluation noise.

@kwdef struct BehaviorCloneConfig
    episodes::Int = 64
    epochs::Int = 100
    batch_size::Int = 256
    lr::Float32 = 1f-3
    # Mixed into the cloned targets so PPO keeps some probability on every action.
    uniform_mix::Float32 = 0.01f0
end

"""
    mixture_initialstate(dists, weights)

Initial-state distribution that samples component `k` with probability `weights[k]`.
"""
function mixture_initialstate(dists, weights)
    length(dists) == length(weights) || throw(ArgumentError("one weight per distribution"))
    all(>=(0), weights) && sum(weights) > 0 || throw(ArgumentError("weights must be nonnegative"))
    cdf = cumsum(Float64.(weights) ./ sum(weights))
    return ImplicitDistribution() do rng
        k = something(findfirst(>=(rand(rng)), cdf), length(cdf))
        return rand(rng, dists[k])
    end
end

function _threaded_map(f, items, shared)
    isempty(items) && return []
    nchunks = min(length(items), Threads.nthreads())
    chunks = collect(Iterators.partition(eachindex(items), cld(length(items), nchunks)))
    tasks = map(chunks) do idxs
        # Each task gets its own copy of planners and accumulators.
        Threads.@spawn begin
            local_shared = deepcopy(shared)
            [f(local_shared, items[i], i) for i in idxs]
        end
    end
    return reduce(vcat, fetch.(tasks))
end

function _selfplay_episode(game, planner, br_player, s, seed; max_steps, gamma)
    Random.seed!(seed)
    rng = Random.MersenneTwister(seed)
    obs = Vector{Float32}[]
    probs = Vector{Float32}[]
    rewards = Float64[]
    for _ in 1:max_steps
        isterminal(game, s) && break
        dist = behavior(planner, s)
        push!(obs, MarkovGames.convert_s(Vector{Float32}, s, game))
        push!(probs, Float32.(dist[br_player].probs))
        a = (rand(rng, dist[1]), rand(rng, dist[2]))
        s, r = @gen(:sp, :r)(game, s, a, rng)
        push!(rewards, player_reward(br_player, r))
    end
    returns = similar(rewards)
    G = 0.0
    for t in reverse(eachindex(rewards))
        G = rewards[t] + gamma * G
        returns[t] = G
    end
    return (; obs, probs, returns=Float32.(returns))
end

"""
    behavior_clone_response(game, planner, br_player; initialstate_dist, max_steps, gamma, config, seed)

Fit a CleanRL-shaped actor to `planner`'s own strategy for `br_player` and a critic to
that seat's discounted return-to-go, both from self-play of `planner`. PPO started from
these networks begins at roughly uᵢ(σᵢ, σ₋ᵢ) instead of a uniform policy.
"""
function behavior_clone_response(
        game::MG,
        planner::Policy,
        br_player::Int;
        initialstate_dist,
        max_steps::Int,
        gamma::Real,
        config::BehaviorCloneConfig = BehaviorCloneConfig(),
        seed::Int = 0,
    )
    state_rng = Random.MersenneTwister(seed)
    starts = [rand(state_rng, initialstate_dist) for _ in 1:config.episodes]
    episodes = _threaded_map(starts, planner) do local_planner, s, i
        _selfplay_episode(game, local_planner, br_player, s, seed + i; max_steps, gamma)
    end
    obs = reduce(hcat, reduce(vcat, [e.obs for e in episodes]))
    probs = reduce(hcat, reduce(vcat, [e.probs for e in episodes]))
    returns = reduce(vcat, [e.returns for e in episodes])

    na = size(probs, 1)
    targets = (1 - config.uniform_mix) .* probs .+ config.uniform_mix / na
    init_rng = Random.MersenneTwister(seed)
    actor, critic = CleanRL.Networks.make_actor_critic(na, size(obs, 1); rng=init_rng)
    actor, critic = Flux.f32(actor), Flux.f32(critic)
    opt_state = Flux.setup(Flux.Adam(config.lr), (actor, critic))
    n = size(obs, 2)
    order = collect(1:n)
    for _ in 1:config.epochs
        Random.shuffle!(init_rng, order)
        for batch in Iterators.partition(order, config.batch_size)
            x, p, g = obs[:, batch], targets[:, batch], returns[batch]
            _, grads = Flux.withgradient((actor, critic)) do (a, c)
                cross_entropy = -mean(sum(p .* Flux.logsoftmax(a(x)); dims=1))
                value_loss = mean((vec(c(x)) .- g) .^ 2)
                cross_entropy + value_loss
            end
            Flux.update!(opt_state, (actor, critic), grads[1])
        end
    end
    kl = mean(sum(targets .* (log.(targets) .- Flux.logsoftmax(actor(obs))); dims=1))
    value_mse = mean((vec(critic(obs)) .- returns) .^ 2)
    return (; actor, critic, samples=n, kl=Float64(kl), value_mse=Float64(value_mse))
end

_seat_joint(response, fixed_seat, br_player::Int) =
    isone(br_player) ? JointPolicy(response, fixed_seat) : JointPolicy(fixed_seat, response)

function _mean_stderr(x)
    m = mean(x)
    return m, length(x) > 1 ? std(x) / sqrt(length(x)) : NaN
end

"""
    seat_rollouts(game, joint_policy, initialstates, br_player; seed, max_steps, accumulators, batch_accumulators)

Run one rollout per initial state, threaded, seeding rollout `k` with `seed + k` so that
two responders evaluated on the same states and seed are paired. Returns the response
player's per-episode discounted rewards and the mean of every scalar batch statistic.
"""
function seat_rollouts(
        game::MG,
        joint_policy::Policy,
        initialstates,
        br_player::Int;
        seed::Int,
        max_steps::Int,
        accumulators,
        batch_accumulators,
    )
    shared = (; joint_policy, accumulators, batch_accumulators)
    episodes = _threaded_map(initialstates, shared) do local_shared, s, i
        Random.seed!(seed + i)
        evaluate_joint_policy(
            game,
            local_shared.joint_policy,
            1;
            initialstates=[s],
            max_steps,
            accumulators=local_shared.accumulators,
            batch_accumulators=local_shared.batch_accumulators,
            show_progress=false,
            parallel=false,
        )
    end
    rewards = [Float64(e.reward[br_player]) for e in episodes]
    metric_keys = filter(keys(first(episodes))) do key
        key ∉ (:n, :max_steps, :eps, :reward) && first(episodes)[key] isa Real
    end
    metrics = NamedTuple{Tuple(metric_keys)}(
        Tuple(mean(Float64(e[key]) for e in episodes) for key in metric_keys),
    )
    return (; rewards, metrics)
end

function _with_fields(config::PPOBestResponseConfig; kwargs...)
    fields = Dict{Symbol,Any}(name => getfield(config, name) for name in fieldnames(PPOBestResponseConfig))
    merge!(fields, Dict{Symbol,Any}(kwargs))
    return PPOBestResponseConfig(; fields...)
end

"""
    ppo_best_response_seat(game, planners, solver, br_player; kwargs...)

Approximate the best response of `br_player` to the `solver` planner in `planners`
(a vector of `name => planner` pairs). `ppo_inits` lists PPO starts: `"bc"` (behavior
cloned from the solver's own seat) and/or `"scratch"`. Every planner's seat and every
policy in `heuristics` (`name => single-seat policy`) is also a candidate. The candidate
with the best mean on `select_runs` held-out rollouts is reported on `eval_runs` fresh
rollouts, paired with the solver's own seat.
"""
function ppo_best_response_seat(
        game::MG,
        planners::AbstractVector{<:Pair},
        solver::AbstractString,
        br_player::Int;
        heuristics::AbstractVector{<:Pair} = Pair{String,Any}[],
        ppo_inits = ("bc",),
        ppo_config::PPOBestResponseConfig,
        bc_config::BehaviorCloneConfig = BehaviorCloneConfig(),
        train_initialstate_dist,
        eval_initialstate_dist,
        select_runs::Int = 200,
        eval_runs::Int = 1000,
        max_steps::Int,
        seed::Int,
        accumulators,
        batch_accumulators,
        model_dir::AbstractString,
    )
    planner_lookup = Dict(planners)
    fixed = planner_lookup[solver]
    fixed_player = MarkovGames.other_player(br_player)
    fixed_seat = ProjectedPlayerPolicy(fixed, fixed_player)
    candidates = Pair{String,Any}[]
    seconds = Dict{String,Float64}()
    saved = Dict{String,Any}()

    bc = nothing
    if "bc" in ppo_inits
        seconds["bc"] = @elapsed bc = behavior_clone_response(
            game, fixed, br_player;
            initialstate_dist=train_initialstate_dist,
            max_steps,
            gamma=ppo_config.gamma,
            config=bc_config,
            seed=seed + 1,
        )
        saved["bc_init"] = (; bc.actor, bc.critic)
        push!(candidates, "bc_init" => ActorPlayerPolicy(game, br_player, bc.actor))
        println("[ppo-br] $(solver) p$(br_player) behavior clone: samples=$(bc.samples) ",
            "kl=$(round(bc.kl; digits=4)) value_mse=$(round(bc.value_mse; digits=4))")
    end
    for init in ppo_inits
        init in ("bc", "scratch") || throw(ArgumentError("unknown PPO init $(init)"))
        name = "response_p$(br_player)__vs_$(solver)_p$(fixed_player)__$(init)"
        warm_start = init == "bc" ? (; initial_actor=bc.actor, initial_critic=bc.critic) : (;)
        println("[ppo-br] training $(name)")
        flush(stdout)
        local response
        seconds["ppo_$(init)"] = @elapsed response = train_ppo_best_response(
            game, fixed, br_player;
            config=_with_fields(ppo_config; name),
            initialstate_dist=train_initialstate_dist,
            warm_start...,
        )
        saved["ppo_$(init)"] = (; response.actor, response.critic)
        push!(candidates, "ppo_$(init)" => ActorPlayerPolicy(game, br_player, response.actor))
        push!(candidates, "ppo_$(init)_greedy" => GreedyActorPlayerPolicy(game, br_player, response.actor))
    end
    for (name, planner) in planners
        push!(candidates, "solver_$(name)" => ProjectedPlayerPolicy(planner, br_player))
    end
    append!(candidates, ("heuristic_$(name)" => policy for (name, policy) in heuristics))

    rollout(policy, states, rollout_seed) = seat_rollouts(
        game, _seat_joint(policy, fixed_seat, br_player), states, br_player;
        seed=rollout_seed, max_steps, accumulators, batch_accumulators,
    )
    select_rng = Random.MersenneTwister(seed + 2)
    eval_rng = Random.MersenneTwister(seed + 3)
    select_states = [rand(select_rng, eval_initialstate_dist) for _ in 1:select_runs]
    eval_states = [rand(eval_rng, eval_initialstate_dist) for _ in 1:eval_runs]

    candidate_rows = NamedTuple[]
    seconds["select"] = @elapsed for (name, policy) in candidates
        result = rollout(policy, select_states, seed + 100_000)
        select_mean, select_stderr = _mean_stderr(result.rewards)
        println("[ppo-br] $(solver) p$(br_player) select $(name): ",
            "$(round(select_mean; digits=4)) ± $(round(select_stderr; digits=4))")
        flush(stdout)
        push!(candidate_rows, (;
            solver, response_player=br_player, candidate=name,
            select_runs, select_mean, select_stderr,
            result.metrics...,
        ))
    end
    selected_index = argmax([row.select_mean for row in candidate_rows])
    selected = first(candidates[selected_index])
    self_name = "solver_$(solver)"
    self_policy = last(only(filter(c -> first(c) == self_name, candidates)))

    local response_result, self_result
    seconds["eval"] = @elapsed begin
        self_result = rollout(self_policy, eval_states, seed + 200_000)
        response_result = selected == self_name ? self_result :
            rollout(last(candidates[selected_index]), eval_states, seed + 200_000)
    end
    response_mean, response_stderr = _mean_stderr(response_result.rewards)
    self_mean, self_stderr = _mean_stderr(self_result.rewards)
    gain_mean, gain_stderr = _mean_stderr(response_result.rewards .- self_result.rewards)
    println("[ppo-br] $(solver) p$(br_player) selected $(selected): ",
        "response=$(round(response_mean; digits=4)) self=$(round(self_mean; digits=4)) ",
        "gain=$(round(gain_mean; digits=4)) ± $(round(gain_stderr; digits=4))")
    flush(stdout)

    mkpath(model_dir)
    model_path = joinpath(model_dir, "responders.jld2")
    jldsave(model_path; responders=saved, selected, seed)

    seat_row = (;
        solver, response_player=br_player, fixed_player, selected,
        eval_runs, response_mean, response_stderr, self_mean, self_stderr,
        gain_mean, gain_stderr,
        bc_samples=isnothing(bc) ? 0 : bc.samples,
        bc_kl=isnothing(bc) ? NaN : bc.kl,
        bc_value_mse=isnothing(bc) ? NaN : bc.value_mse,
        bc_seconds=get(seconds, "bc", 0.0),
        ppo_bc_seconds=get(seconds, "ppo_bc", 0.0),
        ppo_scratch_seconds=get(seconds, "ppo_scratch", 0.0),
        select_seconds=seconds["select"],
        eval_seconds=seconds["eval"],
        seed,
        model_path,
        (Symbol(:response_, key) => value for (key, value) in pairs(response_result.metrics))...,
    )
    return (; candidate_rows, seat_row)
end

function _csv_value(x)
    x isa Nothing && return ""
    x isa AbstractString && return replace(x, ',' => ';', '\n' => ' ')
    return x
end

function write_rows_csv(path::AbstractString, rows)
    columns = Symbol[]
    for row in rows, key in keys(row)
        key in columns || push!(columns, key)
    end
    header = reshape(String.(columns), 1, :)
    data = isempty(rows) ? Matrix{Any}(undef, 0, length(columns)) : reduce(vcat, [
        reshape(Any[_csv_value(get(row, key, nothing)) for key in columns], 1, :)
        for row in rows
    ])
    mkpath(dirname(path))
    writedlm(path, [header; data], ',')
    return path
end

function _solver_summary_rows(seat_rows, solvers)
    rows = NamedTuple[]
    for solver in solvers
        seats = filter(row -> row.solver == solver, seat_rows)
        length(seats) == 2 || continue
        push!(rows, (;
            solver,
            nashconv_lower_bound=sum(row.response_mean for row in seats),
            nashconv_lower_bound_stderr=hypot((row.response_stderr for row in seats)...),
            paired_gain_sum=sum(row.gain_mean for row in seats),
            paired_gain_sum_stderr=hypot((row.gain_stderr for row in seats)...),
            self_play_sum=sum(row.self_mean for row in seats),
            p1_selected=only(filter(row -> row.response_player == 1, seats)).selected,
            p2_selected=only(filter(row -> row.response_player == 2, seats)).selected,
        ))
    end
    return rows
end

"""
    run_ppo_best_response_suite(game, planners; solvers, players, heuristics, output_dir, metadata, seed, seat_kwargs...)

Run `ppo_best_response_seat` for each solver and response player and write
`candidates.csv`, `seats.csv` and `summary.csv` to `output_dir` after every seat.
`heuristics(player)` returns that player's `name => policy` pairs. `metadata` is copied
into every seat row. In `summary.csv`, `nashconv_lower_bound` sums the selected
responders' values and `paired_gain_sum` sums their paired gains over the solver's own
seats; both estimate a lower bound on NashConv in a zero-sum game.
"""
function run_ppo_best_response_suite(
        game::MG,
        planners::AbstractVector{<:Pair};
        solvers = first.(planners),
        players = (1, 2),
        heuristics = player -> Pair{String,Any}[],
        output_dir::AbstractString,
        metadata::NamedTuple = NamedTuple(),
        seed::Int,
        fail_fast::Bool = false,
        ppo_config::PPOBestResponseConfig,
        seat_kwargs...,
    )
    candidate_rows = NamedTuple[]
    seat_rows = NamedTuple[]
    failures = NamedTuple[]
    paths = (
        candidates=joinpath(output_dir, "candidates.csv"),
        seats=joinpath(output_dir, "seats.csv"),
        summary=joinpath(output_dir, "summary.csv"),
        failures=joinpath(output_dir, "failures.csv"),
    )
    for solver in solvers, br_player in players
        try
            result = ppo_best_response_seat(
                game, planners, solver, br_player;
                heuristics=heuristics(br_player),
                ppo_config=_with_fields(ppo_config; log_dir=joinpath(output_dir, "tensorboard")),
                # Seeds depend only on the seat so solvers are compared on the same states.
                seed=seed + 10_000 * br_player,
                model_dir=joinpath(output_dir, solver, "p$(br_player)"),
                seat_kwargs...,
            )
            append!(candidate_rows, result.candidate_rows)
            push!(seat_rows, merge(result.seat_row, metadata))
        catch err
            fail_fast && rethrow()
            @error "PPO best response failed" solver br_player exception=(err, catch_backtrace())
            push!(failures, (; solver, response_player=br_player,
                error=sprint(showerror, err)))
        end
        write_rows_csv(paths.candidates, candidate_rows)
        write_rows_csv(paths.seats, seat_rows)
        write_rows_csv(paths.summary, _solver_summary_rows(seat_rows, solvers))
        isempty(failures) || write_rows_csv(paths.failures, failures)
    end
    return (; candidate_rows, seat_rows, failures, paths)
end
