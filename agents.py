from random import choice
import torch
import os
import pickle

class RandomAgent(object):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def get_action(self):
        # select action at random
        action = choice(self.actions)

        return action

class PolicyAgent(object):
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
            policy = torch.nn.functional.softmax(policy, dim=-1)

            return policy

    def __init__(self, actions, input_size, data_directory_name, alpha=0.001, epsilon=0, gamma=0.99):
        super().__init__()

        # available actions
        self.actions = actions
        self.action_to_index = dict()
        for (index, action) in enumerate(actions):
            self.action_to_index[str(action)] = index

        # data location
        self.data_directory_name = data_directory_name

        # parameters
        self.epsilon = epsilon
        self.gamma = gamma

        # learning
        self.policy_function = self.PolicyFunction(actions, input_size)
        self.optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=alpha)

    def update(self):
        # initialize
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        episode_count = len(os.listdir(self.data_directory_name))

        # read episode data
        for episode_file_name in os.listdir(self.data_directory_name):
            # load episide data
            episode_file_path = os.path.join(self.data_directory_name, episode_file_name)
            with open(episode_file_path, "rb") as episode_data_file: 
                episode_data = pickle.load(episode_data_file)

            total_log_action_probability = torch.tensor(0, dtype=torch.float32, requires_grad=True)
            total_discounted_reward = torch.tensor(0, dtype=torch.float32, requires_grad=True)
            for (index, current_data) in enumerate(episode_data):
                # read data
                state = current_data["state"]
                action = current_data["action"]
                reward = current_data["reward"]

                # calculate loss contribution
                input_data = torch.tensor(state, dtype=torch.float32, requires_grad=True) / 255
                action_probability = self.policy_function(input_data)[self.action_to_index[str(action)]]
                log_action_probability = torch.log(action_probability)
                discounted_reward = torch.tensor((self.gamma**index) * reward, requires_grad=True)
                total_log_action_probability = total_log_action_probability + log_action_probability
                total_discounted_reward = total_discounted_reward + discounted_reward
            
            # add episode loss to total loss
            loss = loss + (total_log_action_probability * total_discounted_reward)

        # calculate mean loss
        loss = loss / episode_count

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state):
        # convert image data to normalized tensor
        data = torch.tensor(state, dtype=torch.float32) / 255

        # get optimal action based on current policy
        policy = self.policy_function(data)
        action = self.actions[torch.argmax(policy)]

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < self.epsilon):
            # take random action
            action = choice(self.actions)

        return action