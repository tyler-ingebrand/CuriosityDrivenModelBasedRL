import numpy as np

from src.CuriousAgent import CuriousAgent
from src.KnownDynamicsAgent import KnownDynamicsAgent
import gym
from env_functions.pendulum import *
from src.UnknownDynamicsAgent import UnknownDynamicsAgent
import matplotlib.pyplot as plt

from src.evaluate import evaluate

## arguments
number_steps = 16_000
known = False
curious = True
test_every = 2_000
number_testing_episodes = 5
curiosity_weight = -0.1 # note negative punishes uncertainty
############
print("Cuda ok? ", torch.cuda.is_available())

# create some env
env = gym.make("Pendulum-v1", render_mode=None)
obs, info = env.reset(seed=1)

test_env = gym.make("Pendulum-v1", render_mode=None)
test_env.reset(seed=123)

# create agent based on that env
if known:
    name = "Known"
    agent = KnownDynamicsAgent(env.action_space,
                               pendelum_reward, pendelum_value, pendelum_dynamics,  # env functions
                               torch.optim.SGD, {"lr": 0.1},  # MPC optimizer type and arguments
                               look_ahead_steps=10, descent_steps=1000)
elif not curious:
    name = "Unknown"
    agent = UnknownDynamicsAgent(env.observation_space, env.action_space,
                                 pendelum_reward, pendelum_value,  # env functions
                                 torch.optim.SGD, {"lr": 0.1},
                                 descent_steps=1000,
                                 look_ahead_steps=10,
                                 train_every_N_steps=500)

else:
    name = "Curious{}".format(curiosity_weight)
    agent = CuriousAgent(env.observation_space, env.action_space,
                         pendelum_reward, pendelum_value,  # env functions
                         torch.optim.SGD, {"lr": 0.1},
                         descent_steps=1000,
                         look_ahead_steps=10,
                         train_every_N_steps=500,
                         curiosity_weight=curiosity_weight)  # negative punishes uncertainty
print("Testing {} Dynamics".format(name))
means, stds, times = [], [], []

# iterate to see what happens
for step in range(number_steps):
    # fetch action from agent given current state
    action = agent(obs)

    # apply action to env
    next_obs, reward, done, truncated, info = env.step(action)

    # learn if we can learn, if not does nothing
    agent.learn(obs, action, next_obs, reward)

    # for our sake
    if step % test_every == 0:
        print("Testing at ", step)
        mean, std_dev = evaluate(agent, test_env, number_testing_episodes, verbose=True)
        means.append(mean)
        stds.append(std_dev)
        times.append(step)

    # reset if needed, otherwise continue
    if done or truncated:
        obs, info = env.reset()
    else:
        obs = next_obs
env.close()

means = np.array(means)
stds = np.array(stds)
times = np.array(times)

with open('data/{}.npy'.format(name), 'wb') as f:
    np.save(f, means)
    np.save(f, stds)
    np.save(f, times)

# display
plt.plot(times, means)
plt.fill_between(times, y1=means+stds, y2=means-stds, color=(0,0,1,0.3))
plt.show()
