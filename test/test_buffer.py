from src.KnownDynamicsAgent import KnownDynamicsAgent
import gym
from env_functions.pendulum import *
from src.replay_buffer import *

# create some env
env = gym.make("Pendulum-v1")
obs, info = env.reset()

# create agent based on that env
buffer = Replay_Buffer(env.observation_space, env.action_space)

# iterate to see what happens
for step in range(2000):
    # fetch action from agent given current state
    action = env.action_space.sample()

    # apply action to env
    next_obs, reward, done, truncated, info = env.step(action)

    buffer.add(obs, action, next_obs)

    # reset if needed, otherwise continue
    if done:
        obs, info = env.reset()
    else:
        obs = next_obs

print("Sampling")
state, action, next_State = buffer.sample(100)
print("State = ", state)
print("action = ", action)
print("next_State = ", next_State)

