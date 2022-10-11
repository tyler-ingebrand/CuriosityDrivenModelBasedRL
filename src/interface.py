
# default interface for an agent. Needs to be callable(Returns action), and learn.
class Agent:
    def __call__(self, state):
        raise "Unimplemented error"

    def learn(self, state, action, next_state, reward):
        pass
