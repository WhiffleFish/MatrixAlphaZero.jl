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
    if hasproperty(info, :mean_regret_loss)
        println("  loss:        total=$(_r(info.mean_loss))  value=$(_r(info.mean_value_loss))  regret=$(_r(info.mean_regret_loss))  strategy=$(_r(info.mean_strategy_loss))")
    elseif hasproperty(info, :mean_policy_loss)
        println("  loss:        total=$(_r(info.mean_loss))  value=$(_r(info.mean_value_loss))  policy=$(_r(info.mean_policy_loss))")
    end
    if hasproperty(info, :mean_grad_norm)
        println("  grad norm:   mean=$(_r3(info.mean_grad_norm))  max=$(_r3(info.max_grad_norm))")
    end
    hasproperty(info, :learning_rate) && println("  learning rate: $(info.learning_rate)")
    if hasproperty(info, :strategy_entropy_p1)
        println("  entropy:     p1=$(_r3(info.strategy_entropy_p1))  p2=$(_r3(info.strategy_entropy_p2))")
        println("  strategy Δkl:  p1=$(_r(info.strategy_kl_p1))  p2=$(_r(info.strategy_kl_p2))")
    end
    if hasproperty(info, :target_strategy_kl_p1)
        target_regret_l2 = hasproperty(info, :target_regret_l2) ? "  target regret l2=$(_r(info.target_regret_l2))" : ""
        println("  oracle↔target kl:  p1=$(_r(info.target_strategy_kl_p1))  p2=$(_r(info.target_strategy_kl_p2))  │  regret mse=$(_r(info.regret_pred_mse))$(target_regret_l2)  value mse=$(_r(info.value_pred_mse))")
    end
    if hasproperty(info, :policy_entropy_p1)
        println("  entropy:     p1=$(_r3(info.policy_entropy_p1))  p2=$(_r3(info.policy_entropy_p2))")
        println("  policy Δkl:  p1=$(_r(info.policy_kl_p1))  p2=$(_r(info.policy_kl_p2))")
    end
    if hasproperty(info, :target_policy_kl_p1)
        target_regret_l2 = hasproperty(info, :target_regret_l2) ? "  target regret l2=$(_r(info.target_regret_l2))" : ""
        regret_mse = hasproperty(info, :regret_pred_mse) ? "  │  regret mse=$(_r(info.regret_pred_mse))$(target_regret_l2)" : ""
        println("  oracle↔target kl:  p1=$(_r(info.target_policy_kl_p1))  p2=$(_r(info.target_policy_kl_p2))$(regret_mse)  value mse=$(_r(info.value_pred_mse))")
    end
    if hasproperty(info, :mean_ep_length)
        println("  self-play:   ep_len=$(_r(info.mean_ep_length, 1))  reward μ=$(_r3(info.mean_reward))  σ=$(_r3(info.reward_std))")
        hasproperty(info, :mean_search_time) && println("  search:      mean=$(_r3(1_000 * info.mean_search_time))ms  total=$(_r2(info.total_search_time))s  n=$(info.search_count)")
    end
    if hasproperty(info, :exploration_epsilon)
        msg = "  exploration: epsilon=$(_r3(info.exploration_epsilon))"
        hasproperty(info, :transfer_tau) && (msg *= "  τ=$(_r3(info.transfer_tau))")
        println(msg)
    end
    if hasproperty(info, :batch_size)
        println("  batch:       size=$(info.batch_size)  steps=$(get(info, :steps_done, info.batch_size))/$(get(info, :max_steps, "?"))")
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
