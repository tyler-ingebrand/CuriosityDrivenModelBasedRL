from src.CuriousAgent import CuriousAgent
from src.KnownDynamicsAgent import KnownDynamicsAgent
import gym
from env_functions.pendulum import *
from src.UnknownDynamicsAgent import UnknownDynamicsAgent
import matplotlib.pyplot as plt

## arguments
number_steps = 100_000
known = False
curious = True
render_after = 50_000
############
print("Cuda ok? ", torch.cuda.is_available())

# create some env
env = gym.make("Pendulum-v1", render_mode=None)
obs, info = env.reset()

# create agent based on that env
if known:
    agent = KnownDynamicsAgent(env.action_space,
                               pendelum_reward, pendelum_value, pendelum_dynamics,  # env functions
                               torch.optim.SGD, {"lr": 0.1},  # MPC optimizer type and arguments
                               look_ahead_steps=10, descent_steps=1000)
elif not curious:
    agent = UnknownDynamicsAgent(env.observation_space, env.action_space,
                                 pendelum_reward, pendelum_value,  # env functions
                                 torch.optim.SGD, {"lr": 0.1},
                                 descent_steps=1000,
                                 look_ahead_steps=10, exploration_steps=50_000,
                                 train_every_N_steps=500)

else:
    agent = CuriousAgent(env.observation_space, env.action_space,
                         pendelum_reward, pendelum_value,  # env functions
                         torch.optim.SGD, {"lr": 0.1},
                         descent_steps=1000,
                         look_ahead_steps=10, exploration_steps=50_000,
                         train_every_N_steps=500,
                         curiosity_weight= -100.0)  # negative punishes uncertainty

# iterate to see what happens
for step in range(number_steps):
    # fetch action from agent given current state
    action = agent(obs)

    # apply action to env
    next_obs, reward, done, truncated, info = env.step(action)

    # learn if we can learn, if not does nothing
    agent.learn(obs, action, next_obs, reward)

    # for our sake
    if step % 1000 == 0 or step > render_after:
        print(step, ": S ", obs, ", A ", action)

    # reset if needed, otherwise continue
    if done or truncated:
        obs, info = env.reset()
    else:
        obs = next_obs

    if step == render_after:
        env = gym.make("Pendulum-v1", render_mode="human")
        obs, info = env.reset()

env.close()
