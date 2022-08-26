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

class PolicyFunction(torch.nn.Module):
    def __init__(self, actions, input_size) -> None:
        super().__init__()
        self.actions = actions

        # neural network layers
        self.convolution_1 = torch.nn.Conv2d(input_size[0], input_size[0], 11, 4)
        self.pooling_1 = torch.nn.MaxPool2d(3, 2)
        self.convolution_2 = torch.nn.Conv2d(input_size[0], input_size[0], 5, 1, 2)
        self.pooling_2 = torch.nn.MaxPool2d(3, 2)
        self.convolution_3 = torch.nn.Conv2d(input_size[0], input_size[0], 3, 1, 1)
        self.convolution_4 = torch.nn.Conv2d(input_size[0], input_size[0], 3, 1, 1)
        self.convolution_5 = torch.nn.Conv2d(input_size[0], input_size[0], 3, 1, 1)
        self.pooling_3 = torch.nn.MaxPool2d(3, 2)
        self.linear_1 = torch.nn.Linear(144, 144)
        self.linear_2 = torch.nn.Linear(144, len(actions))

    def forward(self, state):
        # calculate action probabilities
        policy = self.convolution_1(state)
        policy = torch.nn.functional.relu(policy)
        policy = self.pooling_1(policy)
        policy = self.convolution_2(policy)
        policy = torch.nn.functional.relu(policy)
        policy = self.pooling_2(policy)
        policy = self.convolution_3(policy)
        policy = torch.nn.functional.relu(policy)
        policy = self.convolution_4(policy)
        policy = torch.nn.functional.relu(policy)
        policy = self.convolution_5(policy)
        policy = torch.nn.functional.relu(policy)
        policy = self.pooling_3(policy)
        policy = torch.flatten(policy)
        policy = self.linear_1(policy)
        policy = torch.nn.functional.relu(policy)
        policy = self.linear_2(policy)
        policy = torch.nn.functional.softmax(policy)

        return policy

class PolicyAgent(object):
    def __init__(self, actions, input_size, alpha=0.001, epsilon=0, gamma=0.99, data_buffer_size=100):
        super().__init__()
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.data_buffer = dict()
        self.data_buffer_size = data_buffer_size

        self.policy_function = PolicyFunction(actions, input_size)
        self.optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=alpha)

    def add_to_data_buffer(self, index, state, action, reward, next_state):
        index_data_buffer = self.data_buffer.get(index, [])
        index_data_buffer.append([state, action, reward, next_state])
        self.data_buffer[index] = index_data_buffer

        if(len(self.data_buffer) >= self.data_buffer_size):
            # update model
            self.update()
            
            # empty buffer
            self.data_buffer = []

    def update(self):
        print("todo: update")

    def get_action(self, state):
        # convert image data to normalized tensor
        data = torch.Tensor(state.screen_buffer) / 255

        # get optimal action based on current policy
        policy = self.policy_function(data)
        action = self.actions[torch.argmax(policy)]

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < self.epsilon):
            # take random action
            action = choice(self.actions)

        return action