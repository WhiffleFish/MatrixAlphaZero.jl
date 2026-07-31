## FIXME: DRY --- put this in some common spot so both SNRGame and BeliefGame can use this

function can_see_sat(epc, x_obs, x_sat)
    length(x_obs) == 6 && (x_obs = x_obs[idx1t3])
    length(x_sat) == 6 && (x_sat = x_sat[idx1t3])
    x_sun = sun_position(epc)
    x_moon = moon_position(epc)
    return !(
        earth_occlusion(x_obs, x_sat) ||
        moon_occlusion(x_moon, x_obs, x_sat) ||
        looking_into_sun(x_sun, x_obs, x_sat) || # do we have to look into the sun?
        !is_sat_backlit(x_sun, x_moon, x_sat)
    )
end

const R_MOON = 1_740e3

function closest_lerp_coeff(x1, x2)
    return (dot(x2, x2)-dot(x1,x2)) / (dot(x1, x1) - 2dot(x1, x2) + dot(x2, x2))
end

function looking_into_sun(x_sun, x_obs, x_sat)
    v_so = x_obs - x_sun
    v_ss = x_sat - x_sun

    α = closest_lerp_coeff(v_so, v_ss)
    
    # if α < 0, we're looking in the opposite direction of the sun
    if α < 0
        return false
    else
        d = norm(lerp(v_so, v_ss, α), 2)
        return d < R_SUN
    end
end

# https://math.stackexchange.com/questions/1225494/component-of-a-vector-perpendicular-to-another-vector
function is_sat_backlit(x_sun, x_moon, x_obj)
    ## check if earth is blocking sun
    # x_sun is vec from earth to sun - want sun to earth
    v_se = -x_sun
    # projection of sat pos onto sun vec
    v_proj = (dot(v_se, x_obj) / dot(v_se, v_se)) * v_se
    v_perp = x_obj - v_proj
    d = norm(v_perp)

    earth_blocking = d < R_EARTH

    ## check of moon is blocking sun
    v_ms = x_sun - x_moon
    v_sm = -v_ms
    v_mo = x_obj - x_moon
    v_proj = (dot(v_sm, v_mo) / dot(v_sm, v_sm)) * v_sm
    v_perp = v_mo - v_proj
    d = norm(v_perp)

    moon_blocking = d < R_MOON

    return !earth_blocking && !moon_blocking
end

earth_occlusion(x_obs, x_sat) = norm(closest_point(x_obs, x_sat), 2) ≤ R_EARTH

function moon_occlusion(x_moon, x_obs, x_sat)
    v_mo = x_obs - x_moon
    v_ms = x_sat - x_moon
    return norm(closest_point(v_mo, v_ms), 2) ≤ R_MOON
end

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
