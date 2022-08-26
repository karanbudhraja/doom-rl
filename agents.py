from random import choice
import torch

class RandomAgent(object):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def get_action(self):
        # select action at random
        action = choice(self.actions)

        return action