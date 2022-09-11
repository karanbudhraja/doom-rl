from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from collections import deque
import os

class BaseAgent(ABC):
    def __init__(self, device, action_size, loss_criterion=torch.nn.MSELoss(), memory_size=32, episode_memory_size=10000, batch_size=16, 
                 opt=optim.SGD, lr=0.00025, discount_factor=0.99, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1,
                 load_model=False, log_directory_name="./logs", model_save_file_name="model.pth"):
        self.device = device
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [deque(maxlen=episode_memory_size) for _ in range(memory_size)]
        self.batch_size = batch_size
        self.discount = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.log_directory_name = log_directory_name
        os.makedirs(log_directory_name, exist_ok=True)
        self.model_save_file_path = os.path.join(log_directory_name, model_save_file_name)
        self.criterion = loss_criterion
        self._opt = opt
        self.lr = lr

        if(load_model == True):
            self.load()

    def clear_memory(self, episode_index):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].clear()

    def append_memory(self, episode_index, state, action, reward, next_state, done):
        data_buffer_index = episode_index % self.memory_size
        self.memory[data_buffer_index].append((state, action, reward, next_state, done))

    @abstractmethod
    def get_action(self, state):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self):
        ...