angle_normalize(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π

function sincos_to_angle(sin, cos)

    th = atan(sin/cos)
    # account for tan issues
    if cos < 0.0 && sin > 0.0
        th += pi
    elseif cos < 0.0 && sin < 0.0
        th -= pi    
    end
end

function pendelum_dynamics(s, a)   
    th, thdot = sincos_to_angle(s[2], s[1]), s[3]

    a = clamp(a, -2, 2)
    newthdot =
        thdot +
        (
            -3 * 10 / (2 * 1) * sin(th + pi) +
            3 * a / (1 * 1^2)
        ) * 0.05
    th += newthdot * 0.05
    newthdot = clamp(newthdot, -8, 8)
    return cos(th), sin(th), newthdot  
end
    
function pendelum_reward(s, a, next_s)
    th, thdot = sincos_to_angle(s[2], s[1]), s[3]
    costs = angle_normalize(th)^2 + 0.1 * thdot^2 + 0.001 * a^2
    return -costs             

end
function pendelum_value(s)
    return (s[1] - 1)^2 * 100 # if pendelum = up and down, value = 0. Otherwise value is negative
end