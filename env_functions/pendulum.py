import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def angle_normalize(x) :
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def sincos_to_angle(sin, cos):
    th = torch.atan(sin / cos)
    # account for atan issues
    if cos < 0.0 and sin > 0.0:
        th += torch.pi
    elif cos < 0.0 and sin < 0.0:
        th -= torch.pi
    elif cos == 0.0:
        th = torch.pi/2 * sin
    return th


def pendelum_dynamics(s, a):
    th, thdot = sincos_to_angle(s[1], s[0]), s[2]

    # clamp action if out of bounds
    action = torch.clamp(a[0], -2.0, 2.0)

    # constants taken from pendulum defaults
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05

    # find new theta speed
    newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * action) * dt
    newthdot2 = torch.clamp(newthdot, -8.0, 8.0)

    # find new theta
    newth = th + newthdot * 0.05
    return [torch.cos(newth), torch.sin(newth), newthdot2]


def pendelum_reward(s, a, next_s):
    th, thdot = sincos_to_angle(s[1], s[0]), s[2]
    # costs = torch.tensor([angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (a[0] ** 2)], requires_grad=True)
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2
    return -costs


def pendelum_value(s):
    th, thdot = sincos_to_angle(s[1], s[0]), s[2]
    return -2 * th ** 2.0  # if pendelum = up and down, value = 0. Otherwise value is negative
