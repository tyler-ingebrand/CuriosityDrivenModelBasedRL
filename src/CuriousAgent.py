import numpy
import torch
from torch import nn

from .interface import Agent
from .replay_buffer import Replay_Buffer


class CuriousAgent(Agent):
    def __init__(self,
                 state_space,
                 action_space,  # action space from gym
                 reward_function,  # reward for state action next_state (s,a,s' -> r)
                 value_function,  # Value of state at end of lookadead (s -> V)

                 # MPC optimzation variables
                 optimizer_type,  # which gradient descent method to optimize by. This should be a class, not an object
                 optimizer_kwargs,  # arguments to use to create the optimizer, such as learning rate
                 descent_steps,  # How many times to do gradient descent each time we need an action
                 look_ahead_steps=10,  # how far to lookahead into the future max. Starts at 0, increments to this
                 train_every_N_steps=500,
                 early_termination_difference=1e-4,
                 curiosity_weight=0.0
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Pertain to env
        self.action_space = action_space
        self.state_space = state_space
        self.reward_function = reward_function
        self.value_function = value_function

        # MPC opt variables
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.look_ahead_steps = look_ahead_steps
        self.descent_steps = descent_steps
        self.early_termination_difference = early_termination_difference
        self.warm_start = torch.zeros(self.look_ahead_steps, self.action_space.shape[0]).to(self.device)
        self.curiosity_weight = curiosity_weight

        # General traniing variables
        self.train_every_N_steps = train_every_N_steps
        self.action_low = torch.tensor(action_space.low).to(self.device)
        self.action_high = torch.tensor(action_space.high).to(self.device)
        self.current_step = 0

        # create a replay buffer to store transitions
        self.replay_buffer = Replay_Buffer(state_space, action_space)

        # Dynamics opt variables
        self.model_batch_size = 256
        self.model_number_batches = 10
        self.model_descent_steps = 100
        self._create_dynamics()
        self._create_inverse_and_backwards()
        self.set_exploit(False)

    def _create_dynamics(self):
        # Create dynamics model and optimizer
        ns = self.state_space.shape[0]
        na = self.action_space.shape[0]
        self.forwards_model = nn.Sequential(nn.Linear(ns + na, 64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, ns)).to(self.device)
        self.forwards_optimizer = torch.optim.Adam(self.forwards_model.parameters())

    def _create_inverse_and_backwards(self):
        # Create dynamics model and optimizer
        ns = self.state_space.shape[0]
        na = self.action_space.shape[0]

        # Inverse model, predicts s,s' -> a
        self.inverse_model = nn.Sequential(nn.Linear(ns + ns, 64), nn.ReLU(),
                                           nn.Linear(64, 64), nn.ReLU(),
                                           nn.Linear(64, na)).to(self.device)
        self.inverse_optimizer = torch.optim.Adam(self.inverse_model.parameters())

        # backwards model, predicts s',a -> s
        self.backwards_model = nn.Sequential(nn.Linear(ns + na, 64), nn.ReLU(),
                                             nn.Linear(64, 64), nn.ReLU(),
                                             nn.Linear(64, ns)).to(self.device)
        self.backwards_optimizer = torch.optim.Adam(self.backwards_model.parameters())

    def __call__(self, state):
        if not self.exploit:  # some initial random search phase to collect data fast
            return self.action_space.sample()
        else:
            return self.choose_actions(state)

    def learn(self, obs, action, next_obs, reward):
        self.replay_buffer.add(obs, action, next_obs)
        self.current_step += 1
        if self.current_step % self.train_every_N_steps == 0:
            self.improve_model()

    # helper to choose the actions given an initial state
    def choose_actions(self, state):
        # torch.autograd.set_detect_anomaly(True) # use this if you have an autograd issue
        actions = self.warm_start.clone().detach().requires_grad_(True)
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
                                  self.action_high).clone().detach().requires_grad_(True)

            # Check for early termination. Terminate if action has not changed more than some amount
            if torch.linalg.norm(actions - last_action) < self.early_termination_difference:
                break
            last_action = actions.clone().detach()

        # remember plan for warm start. This makes it find plan faster next time.
        self.warm_start[:-1] = actions[1:]
        self.warm_start[-1] = actions[-1]  # clone last action and assume we reuse it. Works ok, but hackish.

        # Return first action to take
        return actions[0].detach().cpu().numpy()

    # helper to predict the value of a trajectory given an initial state and choice of actions
    # This is differentiable WRT actions
    def predict_value(self, initial_state, actions):
        state = initial_state
        values = 0.0

        # temp_reward = 0.0
        # temp_curiosity = 0.0

        # for each action in our lookahead, simulate transition
        # Account for reward received
        for i, action in enumerate(actions):
            # predict next state
            next_state = self.forwards_model(torch.cat((state, action)))

            # calculate model error
            state_prediction = self.backwards_model(torch.cat((next_state, action)))
            action_prediction = self.inverse_model(torch.cat((state, next_state)))
            error = torch.linalg.norm(state - state_prediction) + torch.linalg.norm(action - action_prediction)

            # temp
            # temp_reward += self.reward_function(state, action, next_state)
            # temp_curiosity += error * self.curiosity_weight

            # add reward and error * curiosity_weight to value, then continue
            values += self.reward_function(state, action, next_state) + error * self.curiosity_weight
            state = next_state

        # account for final state's infinite horizon value
        values += self.value_function(state)

        # print("Curiosity bonus ", temp_curiosity)
        # print("reward bonus ", temp_reward)
        # print("Value bonus ", self.value_function(state))
        return values

    def set_exploit(self, value):
        self.exploit = value
        # for p in self.forwards_model.parameters():
        #     p.requires_grad = not value
        # for p in self.backwards_model.parameters():
        #     p.requires_grad = not value
        # for p in self.inverse_model.parameters():
        #     p.requires_grad = not value
    def improve_model(self):
        for batch in range(self.model_number_batches):
            states, actions, next_states = self.replay_buffer.sample(self.model_batch_size)

            for descent_steps in range(self.model_descent_steps):
                # improve forward model
                self.forwards_model.zero_grad()
                forward_inputs = torch.cat((states, actions), dim=1)
                loss = nn.MSELoss()(self.forwards_model(forward_inputs), next_states)
                loss.backward()
                self.forwards_optimizer.step()

                # improve backwards model
                self.backwards_model.zero_grad()
                backwards_inputs = torch.cat((next_states, actions), dim=1)
                loss = nn.MSELoss()(self.backwards_model(backwards_inputs), states)
                loss.backward()
                self.backwards_optimizer.step()

                # improve inverse model
                self.inverse_model.zero_grad()
                inverse_inputs = torch.cat((states, next_states), dim=1)
                loss = nn.MSELoss()(self.inverse_model(inverse_inputs), actions)
                loss.backward()
                self.inverse_optimizer.step()
