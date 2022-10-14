import math
import torch

def angle_normalize(x) :
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def sincos_to_angle(sin, cos):
    th = math.atan(sin / cos)
    # account for atan issues
    if cos < 0.0 and sin > 0.0:
        th += math.pi
    elif cos < 0.0 and sin < 0.0:
        th -= math.pi
    elif cos == 0.0:
        th = math.pi/2 * sin
    return th


def pendelum_dynamics(s, a):
    th, thdot = sincos_to_angle(s[1], s[0]), s[2]

    # clamp action if out of bounds
    a = torch.clamp(a, -2, 2)

    # constants taken from pendulum defaults
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05

    # find new theta speed
    newthdot = thdot + (3 * g / (2 * l) * math.sin(th) + 3.0 / (m * l ** 2) * a) * dt
    newthdot = torch.clamp(newthdot, -8, 8)

    # find new theta
    th += newthdot * 0.05
    return math.cos(th), math.sin(th), newthdot


def pendelum_reward(s, a, next_s):
    th, thdot = sincos_to_angle(s[1], s[0]), s[2]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (a ** 2)
    return -costs


def pendelum_value(s):
    return -(s[0] - 1) ** 2.0 * 10  # if pendelum = up and down, value = 0. Otherwise value is negative
