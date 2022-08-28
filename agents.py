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

    def __init__(self, actions, input_size, data_directory_name,
                states_file_name, action_policies_file_name, rewards_file_name, next_states_file_name, 
                alpha=0.001, epsilon=0.0, gamma=0.99):
        super().__init__()

        # available actions
        self.actions = actions
        self.action_to_index = dict()
        for (index, action) in enumerate(actions):
            self.action_to_index[str(action)] = index

        # data location
        self.data_directory_name = data_directory_name
        self.states_file_name = states_file_name
        self.action_policies_file_name = action_policies_file_name
        self.rewards_file_name = rewards_file_name
        self.next_states_file_name = next_states_file_name

        # parameters
        self.epsilon = epsilon
        self.gamma = gamma

        # learning
        self.policy_function = self.PolicyFunction(actions, input_size)
        self.optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=alpha)

    def get_policy(self, state, episode_number):
        # convert image data to normalized tensor
        data = torch.tensor(state, dtype=torch.float32)

        # get optimal action based on current policy
        policy = self.policy_function(data)

        # use epsilon-greedy policy
        if(torch.rand((1,1)).item() < self.epsilon):
            # get a random policy
            policy = torch.rand((1,self.number_of_actions))

            # epsilon decay
            self.epsilon = self.epsilon

        return policy

    def update(self):
        # initialize
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        episode_count = len(os.listdir(self.data_directory_name))

        # read episode data
        for episode_directory_name in os.listdir(self.data_directory_name):
            # load episide data
            episode_directory_path = os.path.join(self.data_directory_name, episode_directory_name)
            states = np.load(os.path.join(episode_directory_path, self.states_file_name))
            action_policies = np.load(os.path.join(episode_directory_path, self.action_policies_file_name))
            rewards = np.load(os.path.join(episode_directory_path, self.rewards_file_name))

            # calculate loss
            input_data = torch.tensor(states, dtype=torch.float32, requires_grad=True)
 
            print("in update", input_data.shape)

            action_policiies = self.policy_function(input_data)


            print("jere")
            print(states.shape, x.shape)
            exit(0)
            
            
            
            
            
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
