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
            self.convolution_1 = torch.nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
                                                    torch.nn.ReLU())
            self.convolution_2 = torch.nn.Sequential(torch.nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                                                    torch.nn.ReLU())
            self.convolution_3 = torch.nn.Sequential(torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                                                    torch.nn.ReLU())
            self.convolution_4 = torch.nn.Sequential(torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                                                    torch.nn.ReLU())
            self.maxpool_1 = torch.nn.MaxPool1d(2)
            self.linear_1 = torch.nn.Sequential(torch.nn.Linear(96, 64),
                                                torch.nn.ReLU())
            self.linear_2 = torch.nn.Sequential(torch.nn.Linear(64, len(actions)),
                                                torch.nn.Softmax(dim=-1))

        def forward(self, state):
            # calculate action probabilities
            state = self.convolution_1(state)
            state = self.convolution_2(state)
            state = self.convolution_3(state)
            state = self.convolution_4(state)
            state = torch.permute(state, (0, 2, 1))
            state = self.maxpool_1(state)
            state = torch.flatten(state)
            state = self.linear_1(state)
            policy = self.linear_2(state)

            return policy

    def __init__(self, actions, input_size, data_directory_name, alpha=0.001, epsilon=0.0, gamma=0.99):
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
                input_data = torch.tensor(state, dtype=torch.float32, requires_grad=True)
                action_probability = self.policy_function(input_data)[self.action_to_index[str(action)]]
                log_action_probability = torch.log(action_probability)
                discounted_reward = torch.tensor((self.gamma**index) * reward, requires_grad=True)
                total_log_action_probability = total_log_action_probability + log_action_probability
                total_discounted_reward = total_discounted_reward + discounted_reward
            
            # add episode loss to total loss
            loss = loss + (total_log_action_probability * total_discounted_reward)

        # calculate mean loss
        loss = -1 * loss / episode_count

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state, episode_number):
        # convert image data to normalized tensor
        data = torch.tensor(state, dtype=torch.float32)

        # get optimal action based on current policy
        policy = self.policy_function(data)
        action = self.actions[torch.argmax(policy)]

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < (self.epsilon/episode_number)):
            # take random action
            action = choice(self.actions)

        return action