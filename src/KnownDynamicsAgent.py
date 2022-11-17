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
                 descent_steps,  # How many times to do gradient descent each time we need an action
                 early_termination_difference=1e-4,
                 # Gradient descent terminates early if the difference between 2 steps is less than this amount
                 ):
        self.device = torch.device("cpu")# "cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.reward_function = reward_function
        self.value_function = value_function
        self.dynamics_function = dynamics_function
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.look_ahead_steps = look_ahead_steps
        self.descent_steps = descent_steps
        self.early_termination_difference = early_termination_difference
        self.warm_start = torch.zeros((self.look_ahead_steps, self.action_space.shape[0]), device=self.device)
        self.exploit = False
        self.action_low = torch.tensor(action_space.low, device=self.device)
        self.action_high = torch.tensor(action_space.high, device=self.device)

    def set_exploit(self, value):
        self.exploit = value

    def __call__(self, state):
        if not self.exploit:
            return self.action_space.sample()
        else:
            return self.choose_actions(state)

    # helper to choose the actions given an initial state
    def choose_actions(self, state):
        # torch.autograd.set_detect_anomaly(True) # use this if you have an autograd issue
        actions = self.warm_start.clone().detach().to(self.device).requires_grad_()
        state = torch.tensor(state, requires_grad=False, device=self.device)

        # save last action for early termination checking
        last_action = actions.clone().detach()

        # Loop for some number of optimization steps
        for i in range(self.descent_steps):
            # Generate optimizer. Have to regenerate every step because we clamp the actions.
            opt = self.optimizer_type([actions], **self.optimizer_kwargs)
            opt.zero_grad()

            # predict the value of current actions
            negative_value = -self.predict_value(state, actions)

            # compute gradients
            negative_value.backward()
            opt.step()  # minimizes negative value = maximize value

            # project action back into action space
            actions = torch.clamp(actions,
                                  self.action_low,
                                  self.action_high
                                  ).clone().detach().to(self.device).requires_grad_()

            # Check for early termination. Terminate if action has not changed more than some amount
            if torch.linalg.norm(actions - last_action) < self.early_termination_difference:
                break
            last_action = actions.clone().detach().to(self.device)

        # remember plan for warm start. This makes it find plan faster next time.
        self.warm_start[:-1] = actions[1:]
        self.warm_start[-1] = actions[-1]  # clone last action and assume we reuse it. Works ok, but hackish.

        # Return first action to take
        return actions[0].detach().cpu().numpy()

    # helper to predict the value of a trajectory given an initial state and choice of actions
    # This is differentiable WRT actions
    def predict_value(self, initial_state, actions):
        state = initial_state.clone().detach().to(self.device)
        values = torch.tensor(0.0, device=self.device)

        # for each action in our lookahead, simulate transition
        # Account for reward received
        for i, action in enumerate(actions):
            next_state = self.dynamics_function(state, action)
            reward = self.reward_function(state, action, next_state)
            values = values + reward
            state = next_state

        # account for final state's infinite horizon value
        values += self.value_function(state)
        return values
