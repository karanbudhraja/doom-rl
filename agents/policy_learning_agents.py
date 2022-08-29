import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from torch.distributions import Bernoulli

#
# predict actions
#

class PolicyNet(nn.Module):
    # deep q learning architecture
    def __init__(self, available_actions_count):
        super().__init__()
        self.convolution_1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
                                            nn.BatchNorm2d(8),
                                            nn.ReLU())
        self.convolution_2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                                            nn.BatchNorm2d(8),
                                            nn.ReLU())
        self.convolution_3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                                            nn.BatchNorm2d(8),
                                            nn.ReLU())
        self.convolution_4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                                            nn.BatchNorm2d(16),
                                            nn.ReLU())
        self.linear_policy = nn.Sequential(nn.Linear(192, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, available_actions_count),
                                        nn.Softmax(dim=-1))

    def forward(self, state):
        state = self.convolution_1(state)
        state = self.convolution_2(state)
        state = self.convolution_3(state)
        state = self.convolution_4(state)
        state = state.view(-1, 192)
        policy = self.linear_policy(state)

        return policy

class PolicyLearningAgent:
    def __init__(self, device, action_size, policy_network, loss_criterion, memory_size=32, batch_size=16, 
                 lr=0.00025, discount_factor=0.99, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1,
                 load_model=False, log_directory_name="./logs", model_save_file_name="model.pth"):
        self.device = device
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [deque() for _ in range(memory_size)]
        self.batch_size = batch_size
        self.discount = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        os.makedirs(log_directory_name, exist_ok=True)
        self.model_save_file_path = os.path.join(log_directory_name, model_save_file_name)
        self.criterion = loss_criterion

        if load_model:
            print("Loading model from: ", self.model_save_file_path)
            self.policy_net = torch.load(self.model_save_file_path)
            self.target_net = torch.load(self.model_save_file_path)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.policy_net = policy_network(action_size).to(self.device)
            self.target_net = policy_network(action_size).to(self.device)

        self.opt = optim.SGD(self.policy_net.parameters(), lr=lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.target_net(state)).item()

        return action

    def update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def clear_memory(self, episode_index):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].clear()

    def append_memory(self, episode_index, state, action, reward, next_state, done):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].append((state, action, reward, next_state, done))

    def train(self):
        batches = random.sample(self.memory, self.batch_size)
        
        # process each episode in sample of episodes
        total_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)

            # convert rewards to projected state values
            # discount based on distance from end
            _projected_state_values = np.zeros(rewards.shape)
            for index in np.arange(len(rewards)):
                rewards_subset = rewards[index:]
                discounts = self.discount**np.flip(np.arange(len(rewards_subset)))
                _projected_state_values[index] = np.sum(rewards_subset * discounts)

            next_states = np.stack(batch[:, 3]).astype(float)
            dones = batch[:, 4].astype(bool)
            not_dones = ~dones

            # get log probability of action based on policy network
            action_probabilities = self.policy_net(torch.from_numpy(states).float())
            probability_model = Bernoulli(action_probabilities)            
            mask = np.repeat(np.array(range(action_probabilities.shape[1])).reshape((1, action_probabilities.shape[1])), action_probabilities.shape[0], axis=0)            
            masked_action_probabilities = torch.tensor(np.equal(mask, actions.reshape(-1, 1)), dtype=torch.float32, requires_grad=True)
            log_probability = probability_model.log_prob(masked_action_probabilities)
            log_probability = log_probability * masked_action_probabilities

            # calculate loss
            projected_state_values = torch.tensor(_projected_state_values, dtype=torch.float32, requires_grad=True)
            episode_loss = -1 * torch.sum(log_probability * projected_state_values.reshape((-1, 1)))
            total_loss = total_loss + episode_loss

        average_loss = total_loss / self.batch_size

        # gradient step
        self.opt.zero_grad()
        average_loss.backward()
        self.opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
