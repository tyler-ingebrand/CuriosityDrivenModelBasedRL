import torch

from .interface import Agent


class KnownDynamicsAgent(Agent):
    def __init__(self,
                 action_space,  # action space from gym
                 reward_function,  # reward for state action next_state (s,a,s' -> r)
                 value_function,  # Value of state at end of lookadead (s -> V)
                 dynamics_function,  # Dynamics function (s,a -> s')
                 optimizer_type,  # which gradient descent method to optimize by. This should be a class, not an object
                 optimizer_kwargs,  # arguments to use to create the optimizer, such as learning rate
                 look_ahead_steps,  # how far to lookahead into the future
                 descent_steps  # How many times to do gradient descent each time we need an action
                 ):
        self.action_space = action_space
        self.reward_function = reward_function
        self.value_function = value_function
        self.dynamics_function = dynamics_function
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.look_ahead_steps = look_ahead_steps
        self.descent_steps = descent_steps

    def __call__(self, state):
        return self.choose_actions(state)

    # helper to choose the actions given an initial state
    def choose_actions(self, state):
        # convert state to torch tensor, create random initial actions
        state = torch.tensor(state, requires_grad=False)
        # actions = torch.randn(self.look_ahead_steps, self.action_space.shape[0], requires_grad=True)
        actions = torch.zeros(self.look_ahead_steps, self.action_space.shape[0], requires_grad=True)

        # descent
        for i in range(self.descent_steps):
            # create an optimizer from the class type specified
            opt = self.optimizer_type([actions], **self.optimizer_kwargs)
            opt.zero_grad()

            # predict the value of current actions
            value = -self.predict_value(state, actions)

            # compute gradients
            value.backward()
            opt.step()
            # print(actions)

            # project value back into action space
            # actions = torch.clamp(actions, torch.from_numpy(self.action_space.low), torch.from_numpy(self.action_space.high))
        return actions[0].detach().numpy()

    # helper to predict the value of a trajectory given an initial state and choice of actions
    def predict_value(self, initial_state, actions):
        sum_value = 0
        state = initial_state

        # for each action in our lookahead, account for reward received
        for action in actions:
            next_state = self.dynamics_function(state, action)
            sum_value += self.reward_function(state, action, next_state)
            state = next_state

        # account for final state's infinite horizon value
        print(sum_value)
        sum_value += self.value_function(state)
        print(self.value_function(state))
        print()

        return sum_value
