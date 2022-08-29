import numpy as np

#
# random agent
#

class RandomAgent(object):
    def __init__(self, device, number_of_actions):
        super().__init__()
        self.number_of_actions = number_of_actions
        self.batch_size = np.inf

    def get_action(self, state):
        # get a random action
        action = np.random.randint(self.number_of_actions)

        return action

    def update(self):
        pass

    def append_memory(self, state, action, reward, next_state, done):
        pass
