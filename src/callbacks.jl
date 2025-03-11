call(f, x) = f(x)

call(f::Tuple, x) = foreach(f_i -> call(f_i, x), f)


struct ModelSaveCallback
    path::String
end

iter2string(i::Int, l=4) = lpad(string(i), l, '0')

function (cb::ModelSaveCallback)(info::NamedTuple=(;))
    model = info[:oracle]
    model_state = Flux.state(model)
    n = if hasproperty(info, :iter)
        iter2string(info[:iter])
    else
        ""
    end
    isdir(cb.path) || mkdir(cb.path)
    jldsave(joinpath(cb.path, "oracle" * n * ".jld2"); model_state)  
end
