function train!(sol, oracle, buf)
    (; batchsize, steps_per_iter, train_intensity, optimiser) = sol
    n_batches = (steps_per_iter * train_intensity) ÷ batchsize
    opt_state = Flux.setup(optimiser, oracle)
    losses = Float32[]
    for _ in 1:n_batches
        X, y = get_batch(buf, batchsize)
        ∇θ = Flux.gradient(oracle) do oracle
            l = Flux.mse(dropdims(oracle(X); dims=1), y)
            Flux.Zygote.ignore_derivatives() do
                push!(losses, l)
            end
            return l
        end
        Flux.update!(opt_state, oracle, ∇θ[1])
    end
    return (;losses)
end

function get_batch(buf::Buffer, batchsize::Int)
    idxs = rand(1:length(buf), batchsize)
    return reduce(hcat, buf.s[idxs]), buf.v[idxs]
end
