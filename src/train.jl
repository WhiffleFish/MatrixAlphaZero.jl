function train!(sol, oracle, buf)
    (; batchsize, steps_per_iter, train_intensity, optimiser) = sol
    n_batches = (steps_per_iter * train_intensity) ÷ batchsize
    opt_state = Flux.setup(optimiser, oracle)
    losses = Float32[]
    value_losses = Float32[]
    policy_losses = Float32[]
    for _ in 1:n_batches
        X, v_target, p_target = get_batch(buf, batchsize)
        v_target = prepare_target(oracle.critic, v_target)
        lv, lp = loss(oracle, X, v_target, p_target)
        ∇θ = Flux.gradient(oracle) do oracle
            lv, lp = loss(oracle, X, v_target, p_target)
            l = lv + lp
            Flux.Zygote.ignore_derivatives() do
                push!(losses, l)
                push!(value_losses, lv)
                push!(policy_losses, lp)
            end
            return l
        end
        Flux.update!(opt_state, oracle, ∇θ[1])
    end
    return (; losses, value_losses, policy_losses)
end

function get_batch(buf::Buffer, batchsize::Int)
    idxs = rand(1:length(buf), batchsize)
    p = map(buf.p) do p
        reduce(hcat, p[idxs])
    end
    return reduce(hcat, buf.s[idxs]), buf.v[idxs], p
end

macro modeldir()
    isnothing(__source__.file) && return nothing
    _dirname = dirname(String(__source__.file::Symbol))
    dir = isempty(_dirname) ? pwd() : abspath(_dirname)
    return :(joinpath($dir, "models"))
end
