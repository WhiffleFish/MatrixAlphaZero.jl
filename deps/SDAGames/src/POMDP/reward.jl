function POMDPs.reward(p::SDABMDP, b::BMDPState, a::Int, bp::BMDPState)
    # FIXME: is r_resolve ever NOT zero?
        # fixed in ofer branch somewhere

    # base reward
    r_base = iszero(a) ? 0 : -1

    r_resolve = 0

    if p.rBelFun == "entropy"
        r_bel = -entropy(b.b.discrete.probs)
    elseif p.rBelFun == "entropy_p"
        r_bel = -entropy(bp.b.discrete.probs)
    elseif p.rBelFun == "delta_entropy"
        r_bel = entropy(bp.b.discrete.probs)- entropy(b.b.discrete.probs)
    elseif p.rBelFun == "entropy_efficiency"
        r_bel = 1-entropy(b.b.discrete.probs)/log(length(b.b.discrete.probs))
    
    else
        r_bel = 0.0
        println("No belief reward function defined!")
    end

    if any(x->x>=0.8,b.b.discrete.probs)
        r_resolve = 0
    end
    
    return r_base+r_bel*p.w_belief+r_resolve    
end
