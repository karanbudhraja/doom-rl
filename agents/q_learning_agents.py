import torch
import numpy as np
import random
from agents.base_agent import BaseAgent

#
# predict q values
#

class QLearningAgent(BaseAgent):
    def __init__(self, device, action_size, q_network):
        super().__init__(device, action_size)

        self.q_net = q_network(action_size).to(self.device)
        self.target_net = q_network(action_size).to(self.device)
        self.opt = self._opt(self.q_net.parameters(), lr=self.lr)

    def clear_memory(self, episode_index):
        # keep all data that fits in queue
        pass

    def append_memory(self, episode, state, action, reward, next_state, done):
        # use a single queue for q-learning agents
        self.memory[0].append((state, action, reward, next_state, done))

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
    
    def train(self):
        batch = random.sample(self.memory[0], self.batch_size)
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

    def save(self):
        pass

    def load(self):
        pass
