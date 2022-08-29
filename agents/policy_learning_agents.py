import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

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
                                        nn.Softmax())

    def forward(self, state):
        print("in here")

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
            action = torch.argmax(self.policy_net(state)).item()

        return action

    def update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def append_memory(self, episode_index, state, action, reward, next_state, done):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].append((state, action, reward, next_state, done))

    def train(self):
        batches = random.sample(self.memory, self.batch_size)
        
        # process each episode in sample of episodes
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)
            next_states = np.stack(batch[:, 3]).astype(float)
            dones = batch[:, 4].astype(bool)
            not_dones = ~dones

            # batch indexing
            row_idx = np.arange(self.batch_size)

            with torch.no_grad():
                # get probability of action based on target (old) network
                # the target network will not be trained here
                state_policy_values = self.target_net(torch.from_numpy(states).float()).cpu().data.numpy()
                

                mask = np.repeat(np.array(range(state_policy_values.shape[1])).reshape((1,state_policy_values.shape[1])), state_policy_values.shape[0], axis=0)
                mask = np.equal(mask, actions.reshape(-1,1))

                action_probabilities = state_policy_values * mask


                print(actions, mask, action_probabilities)

                print(state_policy_values.shape, rewards.shape, actions.shape)                
                
                
                # action_probabilities = state_policy_values[actions]

                # for row in row_idx:
                #     print(state_policy_values[row, :])
                #     print(rewards[row])
                #     print(action_probabilities[row, :])



                print("here")
                exit(0)






        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            # get probability of action based on target (old) network
            # the target network will not be trained here
            state_policy_values = self.target_net(torch.from_numpy(states).float()).cpu().data.numpy()



            print(state_policy_values.shape)
            print(actions.shape)

            action_probabilities = state_policy_values[actions]
            print(action_probabilities.shape)



            exit(0)

            pass

            # next_states = torch.from_numpy(next_states).float().to(self.device)
            # idx = row_idx, np.argmax(self.policy_net(next_states).cpu().data.numpy(), 1)
            # next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            # next_state_values = next_state_values[not_dones]

        # # this defines y = r + discount * max_a q(s', a)
        # q_targets = rewards.copy()
        # q_targets[not_dones] += self.discount * next_state_values
        # q_targets = torch.from_numpy(q_targets).float().to(self.device)

        # # this selects only the q values of the actions taken
        # idx = row_idx, actions
        # states = torch.from_numpy(states).float().to(self.device)
        # action_values = self.policy_net(states)[idx].float().to(self.device)

        # self.opt.zero_grad()
        # td_error = self.criterion(q_targets, action_values)
        # td_error.backward()
        # self.opt.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        # else:
        #     self.epsilon = self.epsilon_min
