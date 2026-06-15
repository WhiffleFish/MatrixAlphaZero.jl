function train!(
        sol,
        oracle::FittedRegretModel,
        batch;
        update_epochs = sol.update_epochs,
        num_batches = sol.num_batches,
        optimiser = sol.optimiser,
        opt_state = nothing,
        rng = sol.rng,
        λ = 1f-2
    )
    hasfield(typeof(batch), :regret) || throw(ArgumentError("FittedRegretModel requires regret/strategy training targets"))
    hasfield(typeof(batch), :strategy) || throw(ArgumentError("FittedRegretModel requires regret/strategy training targets"))
    isnothing(opt_state) && (opt_state = Flux.setup(optimiser, oracle))
    update_epochs > 0 || throw(ArgumentError("update_epochs must be positive"))
    num_batches > 0 || throw(ArgumentError("num_batches must be positive"))
    iszero(length(batch.v)) && throw(ArgumentError("cannot train on an empty batch"))
    X, v_target, r_target, s_target = training_arrays(batch)
    v_target = prepare_target(oracle.critic, v_target)
    n_samples = size(X, 2)
    actual_batches = min(num_batches, n_samples)

    losses = Float32[]
    value_losses = Float32[]
    regret_losses = Float32[]
    strategy_losses = Float32[]
    grad_norms = Float32[]
    for _ in 1:update_epochs
        idxs = randperm(rng, n_samples)
        for mb_idxs in minibatches(idxs, actual_batches)
            X_mb = X[:, mb_idxs]
            v_target_mb = target_minibatch(v_target, mb_idxs)
            r_target_mb = map(r -> r[:, mb_idxs], r_target)
            s_target_mb = map(s -> s[:, mb_idxs], s_target)
            ∇θ = Flux.gradient(oracle) do oracle
                lv, lr, ls = loss(oracle, X_mb, v_target_mb, r_target_mb, s_target_mb)
                l = oracle.value_weight * lv + oracle.regret_weight * lr + oracle.strategy_weight * ls
                Flux.Zygote.ignore_derivatives() do
                    push!(losses, l)
                    push!(value_losses, lv)
                    push!(regret_losses, lr)
                    push!(strategy_losses, ls)
                end
                return l
            end
            gn = sqrt(mapreduce(g -> isnothing(g) ? 0f0 : sum(abs2, g), +, Flux.trainables(∇θ[1]); init=0f0))
            push!(grad_norms, gn)
            Flux.update!(opt_state, oracle, ∇θ[1])
        end
    end
    return (; losses, value_losses, regret_losses, strategy_losses, grad_norms)
end

function train!(
        sol,
        oracle::ActorCritic,
        batch;
        update_epochs = sol.update_epochs,
        num_batches = sol.num_batches,
        optimiser = sol.optimiser,
        opt_state = nothing,
        rng = sol.rng,
        λ = 1f-2
    )
    hasfield(typeof(batch), :policy) || throw(ArgumentError("ActorCritic requires policy training targets"))
    isnothing(opt_state) && (opt_state = Flux.setup(optimiser, oracle))
    update_epochs > 0 || throw(ArgumentError("update_epochs must be positive"))
    num_batches > 0 || throw(ArgumentError("num_batches must be positive"))
    iszero(length(batch.v)) && throw(ArgumentError("cannot train on an empty batch"))
    X, v_target, p_target = policy_training_arrays(batch)
    v_target = prepare_target(oracle.critic, v_target)
    n_samples = size(X, 2)
    actual_batches = min(num_batches, n_samples)

    losses = Float32[]
    value_losses = Float32[]
    policy_losses = Float32[]
    grad_norms = Float32[]
    for _ in 1:update_epochs
        idxs = randperm(rng, n_samples)
        for mb_idxs in minibatches(idxs, actual_batches)
            X_mb = X[:, mb_idxs]
            v_target_mb = target_minibatch(v_target, mb_idxs)
            p_target_mb = map(p -> p[:, mb_idxs], p_target)
            ∇θ = Flux.gradient(oracle) do oracle
                lv, lp = loss(oracle, X_mb, v_target_mb, p_target_mb)
                l = oracle.value_weight * lv + oracle.policy_weight * lp
                Flux.Zygote.ignore_derivatives() do
                    push!(losses, l)
                    push!(value_losses, lv)
                    push!(policy_losses, lp)
                end
                return l
            end
            gn = sqrt(mapreduce(g -> isnothing(g) ? 0f0 : sum(abs2, g), +, Flux.trainables(∇θ[1]); init=0f0))
            push!(grad_norms, gn)
            Flux.update!(opt_state, oracle, ∇θ[1])
        end
    end
    return (; losses, value_losses, policy_losses, grad_norms)
end

l2_penalty(x::AbstractArray) = sum(abs2, x) / length(x)

function training_arrays(batch::NamedTuple)
    n_samples = length(batch.v)
    @assert length(batch.s) == n_samples
    @assert length(batch.regret[1]) == length(batch.regret[2]) == n_samples
    @assert length(batch.strategy[1]) == length(batch.strategy[2]) == n_samples
    X = reduce(hcat, batch.s)
    r_target = map(batch.regret) do r
        Float32.(reduce(hcat, r))
    end
    s_target = map(batch.strategy) do s
        raw = Float64.(reduce(hcat, s))
        for col ∈ eachcol(raw)
            normalize_or_uniform!(col)
        end
        Float32.(raw)
    end
    return X, Float32.(batch.v), r_target, s_target
end

function policy_training_arrays(batch::NamedTuple)
    n_samples = length(batch.v)
    @assert length(batch.s) == n_samples
    @assert length(batch.policy[1]) == length(batch.policy[2]) == n_samples
    X = reduce(hcat, batch.s)
    p_target = map(batch.policy) do p
        raw = Float64.(reduce(hcat, p))
        for col ∈ eachcol(raw)
            normalize_or_uniform!(col)
        end
        Float32.(raw)
    end
    return X, Float32.(batch.v), p_target
end

function target_minibatch(target::AbstractVector, idxs)
    return target[idxs]
end

function target_minibatch(target::AbstractMatrix, idxs)
    return target[:, idxs]
end

function minibatches(idxs::AbstractVector{<:Integer}, num_batches::Int)
    n = length(idxs)
    actual_batches = min(num_batches, n)
    return map(1:actual_batches) do batch_idx
        start_idx = div((batch_idx - 1) * n, actual_batches) + 1
        stop_idx = div(batch_idx * n, actual_batches)
        @view idxs[start_idx:stop_idx]
    end
end

macro modeldir(args...)
    isnothing(__source__.file) && return nothing
    _dirname = dirname(String(__source__.file::Symbol))
    dir = isempty(_dirname) ? pwd() : abspath(_dirname)
    return :(joinpath($dir, "models", $(args...))) |> esc
end

macro model(i)
    isnothing(__source__.file) && return nothing
    _dirname = dirname(String(__source__.file::Symbol))
    dir = isempty(_dirname) ? pwd() : abspath(_dirname)

    return :(Flux.loadmodel!(
        AZ.load_oracle($dir),
        joinpath($dir, "models", "oracle" * MatrixAlphaZero.iter2string($i) * ".jld2"))
    ) |> esc
end
