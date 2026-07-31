lerp(x1, x2, α) = α * x1 + (1-α)*x2

function closest_point(x1, x2)
    α = (dot(x2, x2)-dot(x1,x2)) / (dot(x1, x1) - 2dot(x1, x2) + dot(x2, x2))
    if α < 0
        return x1
    elseif α > 1
        return x2
    elseif x1 ≈ x2
        return x1
    else
        return lerp(x1, x2, α)
    end
end

distance(x1, x2) = norm(closest_point(x1, x2), 2)

function occlusion_matrix(X_obs, X_obj, T=size(first(X_obs), 2))
    n_obs = length(X_obs)
    n_obj = length(X_obj)
    O = trues(n_obj, n_obs, T) # is NOT occluded
    for i ∈ 1:n_obj, j ∈ 1:n_obs, t ∈ 1:T
        O[i,j,t] = distance(X_obj[i][1:3,t], X_obs[j][1:3,t]) ≥ R_EARTH
    end
    return O
end
