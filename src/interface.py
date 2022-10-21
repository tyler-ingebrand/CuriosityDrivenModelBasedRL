
# default interface for an agent. Needs to be callable(Returns action), and learn.
class Agent:
    # This should return an action
    def __call__(self, state):
        raise "Unimplemented error"

    # this returns nothing. Any internal models should be updated with the SARS'
    def learn(self, state, action, next_state, reward):
        pass


