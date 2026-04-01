call(f, x) = f(x)

call(f::Tuple, x) = foreach(f_i -> call(f_i, x), f)


struct ModelSaveCallback
    path::String
end

iter2string(i::Int, l=4) = lpad(string(i), l, '0')

struct MetricsCallback
    width::Int
    MetricsCallback(width::Int=60) = new(width)
end

_r(x, d=4) = round(x; digits=d)
_r2(x) = _r(x, 2)
_r3(x) = _r(x, 3)

function (cb::MetricsCallback)(info::NamedTuple)
    hasproperty(info, :iter) || return
    bar = "─" ^ cb.width
    println(bar)
    println("  iter $(info.iter)")
    if hasproperty(info, :mean_loss)
        println("  loss:        total=$(_r(info.mean_loss))  value=$(_r(info.mean_value_loss))  policy=$(_r(info.mean_policy_loss))  v/p=$(_r2(info.value_policy_ratio))")
    end
    if hasproperty(info, :mean_grad_norm)
        println("  grad norm:   mean=$(_r3(info.mean_grad_norm))  max=$(_r3(info.max_grad_norm))")
    end
    if hasproperty(info, :policy_entropy_p1)
        println("  entropy:     p1=$(_r3(info.policy_entropy_p1))  p2=$(_r3(info.policy_entropy_p2))")
        println("  policy Δkl:  p1=$(_r(info.policy_kl_p1))  p2=$(_r(info.policy_kl_p2))")
    end
    if hasproperty(info, :search_oracle_kl_p1)
        println("  oracle↔search kl:  p1=$(_r(info.search_oracle_kl_p1))  p2=$(_r(info.search_oracle_kl_p2))  │  value mse=$(_r(info.value_pred_mse))")
    end
    if hasproperty(info, :mean_ep_length)
        println("  self-play:   ep_len=$(_r(info.mean_ep_length, 1))  reward μ=$(_r3(info.mean_reward))  σ=$(_r3(info.reward_std))")
    end
    if hasproperty(info, :buffer_size)
        println("  buffer:      size=$(info.buffer_size)  turnover=$(_r3(info.buffer_turnover))")
    end
    println(bar)
end

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
