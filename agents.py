from random import choice
import torch
import os
import numpy as np

class RandomAgent(object):
    def __init__(self, number_of_actions):
        super().__init__()
        self.number_of_actions = number_of_actions

    def get_policy(self, state, episode_number):
        # get a random policy
        policy = torch.rand((1,self.number_of_actions))

        return policy

    def update(self):
        # generate random number
        loss = torch.rand((1,1))

        return loss.item()

class PolicyAgent(object):
    class PolicyFunction(torch.nn.Module):
        def __init__(self, input_size, number_of_actions) -> None:
            super().__init__()

            # neural network layers
            self.convolution_1 = torch.nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
                                                    torch.nn.BatchNorm2d(8),
                                                    torch.nn.ReLU())
            self.convolution_2 = torch.nn.Sequential(torch.nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                                                    torch.nn.BatchNorm2d(8),
                                                    torch.nn.ReLU())
            self.convolution_3 = torch.nn.Sequential(torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                                                    torch.nn.BatchNorm2d(8),
                                                    torch.nn.ReLU())
            self.convolution_4 = torch.nn.Sequential(torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                                                    torch.nn.BatchNorm2d(16),
                                                    torch.nn.ReLU())
            self.linear_1 = torch.nn.Sequential(torch.nn.Linear(192, 64),
                                                torch.nn.ReLU())
            self.linear_2 = torch.nn.Sequential(torch.nn.Linear(64, number_of_actions),
                                                torch.nn.Softmax(dim=-1))

        def forward(self, state):
            # calculate action probabilities
            state = self.convolution_1(state)
            state = self.convolution_2(state)
            state = self.convolution_3(state)
            state = self.convolution_4(state)
            state = state.reshape((state.shape[0], -1))
            state = self.linear_1(state)
            policy = self.linear_2(state)

            return policy

    def __init__(self, input_size, number_of_actions, data_directory_name,
                states_file_name, action_policies_file_name, rewards_file_name, next_states_file_name, 
                alpha=0.001, epsilon=0.5, epsilon_decay=0.99, gamma=1):
        super().__init__()

        # available actions
        self.number_of_actions = number_of_actions

        # data location
        self.data_directory_name = data_directory_name
        self.states_file_name = states_file_name
        self.action_policies_file_name = action_policies_file_name
        self.rewards_file_name = rewards_file_name
        self.next_states_file_name = next_states_file_name

        # parameters
        self.epsilon = epsilon
        self.episilon_decay = epsilon_decay
        self.gamma = gamma

        # learning
        self.policy_function = self.PolicyFunction(input_size, number_of_actions)
        self.optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=alpha)
        self.policy_function.eval()

    def get_policy(self, state, episode_number):
        # convert image data to normalized tensor
        data = torch.tensor(state, dtype=torch.float32)

        # get optimal action based on current policy
        policy = self.policy_function(data)

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < (self.epsilon * (self.episilon_decay**episode_number))):
            # get a random policy
            policy = torch.rand((1, self.number_of_actions))

        return policy

    def update(self):
        # initialize
        self.policy_function.train()
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        episode_count = len(os.listdir(self.data_directory_name))

        # read episode data
        episode_directory_names = os.listdir(self.data_directory_name)
        episode_directory_names.sort()
        for episode_directory_name in episode_directory_names:
            # load episide data
            episode_directory_path = os.path.join(self.data_directory_name, episode_directory_name)
            states = torch.tensor(np.load(os.path.join(episode_directory_path, self.states_file_name)),
                                    dtype=torch.float32, requires_grad=True)
            observed_action_policies = torch.tensor(np.load(os.path.join(episode_directory_path, self.action_policies_file_name)),
                                    dtype=torch.float32, requires_grad=True)
            rewards = torch.tensor(np.load(os.path.join(episode_directory_path, self.rewards_file_name)),
                                    dtype=torch.float32, requires_grad=True)
            reward_discounts = torch.tensor(np.array([self.gamma**index for index in range(rewards.shape[0])]),
                                            dtype=torch.float32, requires_grad=True)
            total_discounted_reward = torch.sum(rewards * reward_discounts)

            # calculate loss
            # multiply by -1 for gradient descent
            # action_policies = self.policy_function(states) 
            # log_policy_values = torch.log(action_policies) * observed_action_policies
            # total_log_policy = torch.sum(log_policy_values)
            # episode_loss = -1 * total_log_policy * total_discounted_reward
            episode_loss = -1 * total_discounted_reward
            loss = loss + episode_loss

        # calculate mean loss
        loss = loss / episode_count

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_function.eval()

        return loss.item()
