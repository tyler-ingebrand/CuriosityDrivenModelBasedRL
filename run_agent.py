from src.KnownDynamicsAgent import KnownDynamicsAgent
import gym
from env_functions.pendulum import *

# create some env
env = gym.make("Pendulum-v1", render_mode="human")
obs, info = env.reset()

# create agent based on that env
agent = KnownDynamicsAgent(env.action_space, pendelum_reward, pendelum_value, pendelum_dynamics, torch.optim.SGD, {"lr": 0.1}, 10, 1000)

# iterate to see what happens
for step in range(200):
    # fetch action from agent given current state
    action = agent(obs)

    # apply action to env
    next_obs, reward, done, truncated, info = env.step(action)

    # learn if we can learn, if not does nothing
    agent.learn(obs, action, next_obs, reward)

    # for our sake
    env.render()
    print("S ", obs, ", A ", action)

    # reset if needed, otherwise continue
    if done:
        obs, info = env.reset()
    else:
        obs = next_obs

env.close()