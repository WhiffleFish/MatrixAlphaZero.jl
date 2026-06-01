struct FunctionPlayerPolicy{G,F} <: Policy
    game::G
    player::Int
    f::F
end

FunctionPlayerPolicy(f, game::MG, player::Int) = FunctionPlayerPolicy(game, player, f)

function MarkovGames.behavior(pol::FunctionPlayerPolicy, s)
    b = pol.f(pol.game, s)
    if b isa AbstractVector
        return SparseCat(actions(pol.game)[pol.player], b)
    else
        return b
    end
end

mutable struct StdErrResult{K,N,T}
    mean::T
    m2::T
    count::Int
end

function StdErrResult(key::Symbol; name::Symbol=Symbol(:stderr_, key), init=0.0)
    z = zero(init)
    return StdErrResult{key,name,typeof(z)}(z, z, 0)
end

function MarkovGames.reset!(stat::StdErrResult{K,N,T}) where {K,N,T}
    stat.mean = zero(T)
    stat.m2 = zero(T)
    stat.count = 0
    return stat
end

function MarkovGames.observe_sim!(stat::StdErrResult{K}, result) where K
    x = result[K]
    stat.count += 1
    if isone(stat.count)
        stat.mean = x
        stat.m2 = zero(stat.m2)
    else
        δ = x .- stat.mean
        stat.mean = stat.mean .+ δ ./ stat.count
        δ2 = x .- stat.mean
        stat.m2 = stat.m2 .+ δ .* δ2
    end
    return stat
end

function MarkovGames.batch_result(stat::StdErrResult{K,N}) where {K,N}
    stderr = stat.count < 2 ? zero(stat.m2) : sqrt.(stat.m2 ./ ((stat.count - 1) * stat.count))
    return NamedTuple{(N,)}((stderr,))
end

function _default_rollout_accumulators()
    return (StepCount(),)
end

function _default_batch_accumulators(game::MG)
    return (
        MeanResult(:steps; name=:mean_steps),
        StdErrResult(:reward; name=:stderr_reward, init=zero(MarkovGames.reward_type(game))),
    )
end

function evaluate_joint_policy(
        game::MG,
        joint_policy::Policy,
        n::Integer;
        accumulators=nothing,
        batch_accumulators=nothing,
        max_steps::Int=typemax(Int),
        eps::Float64=0.0,
        initialstates=nothing,
        metadata=NamedTuple(),
        pool=nothing,
        show_progress::Bool=true,
        proc_warn::Bool=false,
    )
    rollout_accumulators = isnothing(accumulators) ? _default_rollout_accumulators() : accumulators
    batch_stats = isnothing(batch_accumulators) ? _default_batch_accumulators(game) : batch_accumulators
    result = run_stats_parallel(
        game,
        joint_policy,
        n;
        accumulators = rollout_accumulators,
        batch_accumulators = batch_stats,
        max_steps,
        eps,
        initialstates,
        metadata,
        pool,
        show_progress,
        proc_warn,
    )
    return merge((; n=Int(n), max_steps, eps), result)
end
