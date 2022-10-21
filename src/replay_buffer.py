import numpy as np
import torch
class Replay_Buffer:
    def __init__(self, state_space, action_space, buffer_size=100_000):
        self.state_space = state_space
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.states = torch.zeros((self.buffer_size, state_space.shape[0]), device=self.device)
        self.actions = torch.zeros((self.buffer_size, action_space.shape[0]), device=self.device)
        self.next_states = torch.zeros((self.buffer_size, state_space.shape[0]), device=self.device)

        self.pos = 0

    # add data to our buffer
    def add(self, state, action, next_state):
        self.states[self.pos] = torch.tensor(state)
        self.actions[self.pos] = torch.tensor(action)
        self.next_states[self.pos] = torch.tensor(next_state)
        self.pos += 1
        if self.pos == self.buffer_size:
            raise Exception("Out of space")

    # sample a random batch of data from the dataset
    def sample(self, batch_size):
        batch_inds = np.random.randint(0, self.pos, size=batch_size)
        data = (self.states[batch_inds], self.actions[batch_inds], self.next_states[batch_inds])
        return data
