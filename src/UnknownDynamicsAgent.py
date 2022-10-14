import numpy
import torch
from torch import nn

from .interface import Agent
from .replay_buffer import Replay_Buffer


class UnknownDynamicsAgent(Agent):
    def __init__(self,
                 state_space,
                 action_space,  # action space from gym
                 reward_function,  # reward for state action next_state (s,a,s' -> r)
                 value_function,  # Value of state at end of lookadead (s -> V)

                 # MPC optimzation variables
                 optimizer_type,  # which gradient descent method to optimize by. This should be a class, not an object
                 optimizer_kwargs,  # arguments to use to create the optimizer, such as learning rate
                 look_ahead_steps,  # how far to lookahead into the future
                 descent_steps  # How many times to do gradient descent each time we need an action
                 ):
        self.action_space = action_space
        self.state_space = state_space
        self.reward_function = reward_function
        self.value_function = value_function
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.look_ahead_steps = look_ahead_steps
        self.descent_steps = descent_steps

        # create a replay buffer
        self.replay_buffer = Replay_Buffer(state_space, action_space)

        # Create dynamics model and optimizer
        input_size = state_space.shape[0] + action_space.shape[0]
        output_size = state_space.shape[0]
        self.dynamics_approximator = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                                                   nn.Linear(64, 64), nn.ReLU(),
                                                   nn.Linear(64, output_size))
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_approximator.parameters())

        # how often to improve model
        self.train_every_N_steps = 500
        self.current_step = 0

    def __call__(self, state):
        return self.choose_actions(state)

    def learn(self, obs, action, next_obs, reward):
        self.replay_buffer.add(obs, action, next_obs)
        self.current_step += 1
        if self.current_step % self.train_every_N_steps == 0:
            self.improve_model()

    # helper to choose the actions given an initial state
    def choose_actions(self, state):
        # convert state to torch tensor, create random initial actions
        state = torch.tensor(state)
        actions = torch.randn(self.look_ahead_steps, self.action_space.shape[0], requires_grad=True)

        # create an optimizer from the class type specified
        opt = self.optimizer_type([actions], **self.optimizer_kwargs)

        # descent
        for i in range(self.descent_steps):
            opt.zero_grad()

            # predict the value of current actions
            value = -self.predict_value(state, actions)

            # compute gradients
            # value.backward()
            opt.step()

            # project value back into action space
            actions = torch.clamp(actions, torch.from_numpy(self.action_space.low),
                                  torch.from_numpy(self.action_space.high))
        return actions[0].detach().numpy()

    # helper to predict the value of a trajectory given an initial state and choice of actions
    def predict_value(self, initial_state, actions):
        sum_value = 0
        state = initial_state

        # for each action in our lookahead, account for reward received
        for action in actions:
            next_state = self.dynamics_approximator(torch.hstack((state, action)))
            sum_value += self.reward_function(state, action, next_state)
            state = next_state

        # account for final state's infinite horizon value
        sum_value += self.value_function(state)

        return sum_value

    def improve_model(self):

        for batch in range(10):
            states, actions, next_states = self.replay_buffer.sample(256)

            for descent_steps in range(100):
                forward_inputs = numpy.hcat(states, actions)
                loss = nn.MSELoss()(self.dynamics_approximator(forward_inputs), next_states)
                self.dynamics_approximator.zero_grad()
                loss.backward()
                self.dynamics_optimizer.step()
