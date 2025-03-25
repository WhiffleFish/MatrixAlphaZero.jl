function train!(sol, oracle, buf)
    (; batchsize, steps_per_iter, train_intensity, optimiser) = sol
    n_batches = (steps_per_iter * train_intensity) ÷ batchsize
    opt_state = Flux.setup(optimiser, oracle)
    losses = Float32[]
    value_losses = Float32[]
    policy_losses = Float32[]
    for _ in 1:n_batches
        X, v_target, p_target = get_batch(buf, batchsize)
        ∇θ = Flux.gradient(oracle) do oracle
            v, p = oracle(X; logits=true)
            lv = Flux.Losses.huber_loss(dropdims(v; dims=1), v_target)
            lp = mapreduce(+, p, p_target) do p_i, p_target_i
                Flux.Losses.logitcrossentropy(p_i, p_target_i)
            end
            loss = lv + lp
            Flux.Zygote.ignore_derivatives() do
                push!(losses, loss)
                push!(value_losses, lv)
                push!(policy_losses, lp)
            end
            return loss
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
