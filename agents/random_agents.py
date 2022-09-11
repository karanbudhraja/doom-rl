import numpy as np
from agents.base_agent import BaseAgent

#
# random agent
#

class RandomAgent(BaseAgent):
    def __init__(self, device, action_size):
        super().__init__(device, action_size)

    def get_action(self, state):
        # get a random action
        action = np.random.randint(self.action_size)

        return action

    def update(self):
        pass
    
    def train(self):
        return np.inf

    def save(self):
        pass

    def load(self):
        pass
