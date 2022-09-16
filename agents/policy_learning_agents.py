import torch
import numpy as np

import random
from torch.distributions import Categorical
from agents.base_agent import BaseAgent

#
# predict actions
#

class REINFORCEAgent(BaseAgent):
    def __init__(self, device, action_size, policy_network):
        super().__init__(device, action_size)

        self.policy_net = policy_network(action_size).to(self.device)
        self.target_net = policy_network(action_size).to(self.device)
        self.opt = self._opt(self.policy_net.parameters(), lr=self.lr)

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

    def train(self):
        # select from non-empty memory
        counts = np.array([len(x) > 0 for x in self.memory]).astype(int)
        batches = random.choices(self.memory, weights=counts, k=self.batch_size)

        # process each episode in sample of episodes
        episode_losses = []
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)

            # calculate reward to go (estimated q values)
            # discount based on distance from end
            g_values = np.zeros(rewards.shape)
            for index in np.arange(len(rewards)):
                rewards_subset = rewards[index:]
                discounts = self.discount**np.flip(np.arange(len(rewards_subset)))
                g_values[index] = np.dot(rewards_subset, discounts)

            # get log probability of action based on policy and target networks
            # calulate importance weight
            _actions = torch.from_numpy(actions).float().to(self.device)
            policy_action_probabilities = self.policy_net(torch.from_numpy(states).float().to(self.device))
            policy_probability_model = Categorical(policy_action_probabilities)
            policy_log_probability = policy_probability_model.log_prob(_actions).sum()

            # calculate loss
            _g_values = torch.tensor(g_values, requires_grad=True).float().to(self.device)
            episode_loss = -1 * policy_log_probability * _g_values
            episode_losses.append(episode_loss.sum())

        # gradient step
        self.opt.zero_grad()
        batch_loss = torch.stack(episode_losses).mean()
        batch_loss.backward()
        self.opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return batch_loss.clone().item()

    def save(self):
        pass

    def load(self):
        pass

class QACAgent(BaseAgent):
    def __init__(self, device, action_size, q_network, policy_network):
        super().__init__(device, action_size)

        self.current_q_net = q_network(action_size).to(self.device)
        self.current_policy_net = policy_network(action_size).to(self.device)
        self.target_q_net = q_network(action_size).to(self.device)
        self.target_policy_net = policy_network(action_size).to(self.device)
        self.q_opt = self._opt(self.current_q_net.parameters(), lr=self.lr)
        self.policy_opt = self._opt(self.current_policy_net.parameters(), lr=self.lr)

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

    def train(self):
        # select from non-empty memory
        counts = np.array([len(x) > 0 for x in self.memory]).astype(int)
        batches = random.choices(self.memory, weights=counts, k=self.batch_size)






        # TODO: use q net instead of g values and calculate td loss for trianing q net





        # process each episode in sample of episodes
        episode_losses = []
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)

            # convert rewards to projected state values
            # discount based on distance from end
            g_values = np.zeros(rewards.shape)
            for index in np.arange(len(rewards)):
                rewards_subset = rewards[index:]
                discounts = self.discount**np.flip(np.arange(len(rewards_subset)))
                g_values[index] = np.dot(rewards_subset, discounts)

            # get log probability of action based on policy and target networks
            # calulate importance weight
            _actions = torch.from_numpy(actions).float().to(self.device)
            policy_action_probabilities = self.policy_net(torch.from_numpy(states).float().to(self.device))
            policy_probability_model = Categorical(policy_action_probabilities)                  
            policy_log_probability = policy_probability_model.log_prob(_actions)

            # calculate loss
            # _g_values = torch.from_numpy(g_values).float().to(self.device)
            _g_values = torch.tensor(g_values, requires_grad=True).float().to(self.device)
            episode_loss = -1 * policy_log_probability * _g_values
            episode_losses.append(episode_loss.sum())

        # gradient step
        self.opt.zero_grad()
        batch_loss = torch.stack(episode_losses).mean()
        batch_loss.backward()
        self.opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return batch_loss.clone().item()

    def save(self):
        pass

    def load(self):
        pass
