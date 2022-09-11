import torch
import numpy as np

import torch.optim as optim
import random
from collections import deque
import os

#
# predict q values
#

class QLearningAgent:
    def __init__(self, device, action_size, q_network, loss_criterion, memory_size=10000, batch_size=64, 
                 lr=0.00025, discount_factor=0.99, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1,
                 load_model=False, log_directory_name="./logs", model_save_file_name="model.pth"):
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.discount = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.log_directory_name = log_directory_name
        os.makedirs(log_directory_name, exist_ok=True)
        self.model_save_file_path = os.path.join(log_directory_name, model_save_file_name)
        self.criterion = loss_criterion

        if load_model:
            print("Loading model from: ", self.model_save_file_path)
            self.q_net = torch.load(self.model_save_file_path)
            self.target_net = torch.load(self.model_save_file_path)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = q_network(action_size).to(self.device)
            self.target_net = q_network(action_size).to(self.device)

        self.opt = optim.SGD(self.q_net.parameters(), lr=lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.q_net(state)).item()

        return action

    def update(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        # batch indexing
        row_idx = np.arange(self.batch_size)

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(self.device)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(self.device)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(self.device)
        action_values = self.q_net(states)[idx].float().to(self.device)

        # gradient step
        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return td_error.clone().item()
