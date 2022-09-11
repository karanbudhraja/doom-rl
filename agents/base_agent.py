from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from collections import deque
import os

class BaseAgent(ABC):
    def __init__(self, device, action_size, policy_network=None, loss_criterion=None, memory_size=32, batch_size=16, 
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
        self.log_directory_name = log_directory_name
        os.makedirs(log_directory_name, exist_ok=True)
        self.model_save_file_path = os.path.join(log_directory_name, model_save_file_name)
        self.criterion = loss_criterion
        self.opt = None

        if(policy_network != None):
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

    def clear_memory(self, episode_index):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].clear()

    def append_memory(self, episode_index, state, action, reward, next_state, done):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].append((state, action, reward, next_state, done))

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        pass