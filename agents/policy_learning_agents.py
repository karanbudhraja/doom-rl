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
            _actions = torch.from_numpy(actions).to(self.device).reshape(-1, 1)
            policy_action_probabilities = self.policy_net(torch.from_numpy(states).float().to(self.device))
            policy_probability_model = Categorical(policy_action_probabilities)
            policy_log_probability = policy_probability_model.log_prob(_actions.squeeze())

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
    def __init__(self, device, action_size, policy_network, q_network):
        super().__init__(device, action_size)

        self.current_actor_net = policy_network(action_size).to(self.device)
        self.current_critic_net = q_network(action_size).to(self.device)
        self.target_actor_net = policy_network(action_size).to(self.device)
        self.target_critic_net = q_network(action_size).to(self.device)
        self.actor_opt = self._opt(self.current_actor_net.parameters(), lr=self.lr)
        self.critic_opt = self._opt(self.current_critic_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.target_actor_net(state)).item()

        return action

    def update(self):
        self.target_actor_net.load_state_dict(self.current_actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.current_critic_net.state_dict())

    def train(self):
        # select from non-empty memory
        counts = np.array([len(x) > 0 for x in self.memory]).astype(int)
        batches = random.choices(self.memory, weights=counts, k=self.batch_size)

        # process each episode in sample of episodes
        episode_actor_losses = []
        episode_critic_losses = []
        total_sample_count = 0
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)
            next_states = np.stack(batch[:, 3]).astype(float)
            dones = batch[:, 4].astype(bool)
            not_dones = ~dones

            #
            # actor loss
            #

            # get log probability of action based on policy and target networks
            # calulate importance weight
            _states = torch.from_numpy(states).float().to(self.device)
            _actions = torch.from_numpy(actions).to(self.device).reshape(-1, 1)
            q_values = torch.gather(self.current_critic_net(_states), 1, _actions)
            policy_action_probabilities = self.current_actor_net(_states)
            policy_probability_model = Categorical(policy_action_probabilities)
            policy_log_probability = policy_probability_model.log_prob(_actions.squeeze())

            # calculate loss
            episode_loss = -1 * policy_log_probability * q_values
            episode_actor_losses.append(episode_loss.sum())

            #
            # critic loss
            #
    
            # batch indexing
            episode_sample_count = len(rewards)
            total_sample_count += episode_sample_count
            row_idx = np.arange(episode_sample_count)

            # value of the next states with double q learning
            # see https://arxiv.org/abs/1509.06461 for more information on double q learning
            with torch.no_grad():
                next_states = torch.from_numpy(next_states).float().to(self.device)
                idx = row_idx, np.argmax(self.current_critic_net(next_states).cpu().data.numpy(), 1)
                next_state_values = self.target_critic_net(next_states).cpu().data.numpy()[idx]
                next_state_values = next_state_values[not_dones]

            # this defines y = r + discount * max_a q(s', a)
            q_targets = rewards.copy()
            q_targets[not_dones] += self.discount * next_state_values
            q_targets = torch.from_numpy(q_targets).float().to(self.device)

            # this selects only the q values of the actions taken
            idx = row_idx, actions
            states = torch.from_numpy(states).float().to(self.device)
            action_values = self.current_critic_net(states)[idx].float().to(self.device)

            # calculate loss
            # we will take mean later, so calculate raw sum for now
            td_error = self.criterion(q_targets, action_values) * episode_sample_count
            episode_critic_losses.append(td_error)

        # actor gradient step
        self.actor_opt.zero_grad()
        batch_actor_loss = torch.stack(episode_actor_losses).mean()
        batch_actor_loss.backward()
        self.actor_opt.step()

        # critic gradient step
        self.critic_opt.zero_grad()
        batch_critic_loss = torch.stack(episode_critic_losses).sum() / total_sample_count
        batch_critic_loss.backward()
        self.critic_opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return batch_actor_loss.clone().item()

    def save(self):
        pass

    def load(self):
        pass

class AACAgent(BaseAgent):
    def __init__(self, device, action_size, policy_network, q_network):
        super().__init__(device, action_size)

        self.current_actor_net = policy_network(action_size).to(self.device)
        self.current_critic_net = q_network(action_size).to(self.device)
        self.target_actor_net = policy_network(action_size).to(self.device)
        self.target_critic_net = q_network(action_size).to(self.device)
        self.actor_opt = self._opt(self.current_actor_net.parameters(), lr=self.lr)
        self.critic_opt = self._opt(self.current_critic_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.target_actor_net(state)).item()

        return action

    def update(self):
        self.target_actor_net.load_state_dict(self.current_actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.current_critic_net.state_dict())

    def train(self):
        # select from non-empty memory
        counts = np.array([len(x) > 0 for x in self.memory]).astype(int)
        batches = random.choices(self.memory, weights=counts, k=self.batch_size)

        # process each episode in sample of episodes
        episode_actor_losses = []
        episode_critic_losses = []
        total_sample_count = 0
        for batch in batches:
            batch = np.array(batch, dtype=object)
            states = np.stack(batch[:, 0]).astype(float)
            actions = batch[:, 1].astype(int)
            rewards = batch[:, 2].astype(float)
            next_states = np.stack(batch[:, 3]).astype(float)
            dones = batch[:, 4].astype(bool)
            not_dones = ~dones

            #
            # actor loss
            #

            # get log probability of action based on policy and target networks
            # calulate importance weight
            _states = torch.from_numpy(states).float().to(self.device)
            _actions = torch.from_numpy(actions).to(self.device).reshape(-1, 1)
            q_values = torch.gather(self.current_critic_net(_states), 1, _actions)
            policy_action_probabilities = self.current_actor_net(_states)
            policy_probability_model = Categorical(policy_action_probabilities)
            policy_log_probability = policy_probability_model.log_prob(_actions.squeeze())

            # calculate loss
            episode_loss = -1 * policy_log_probability * q_values
            episode_actor_losses.append(episode_loss.sum())

            #
            # critic loss
            #
    
            # TODO

            # batch indexing
            episode_sample_count = len(rewards)
            total_sample_count += episode_sample_count
            row_idx = np.arange(episode_sample_count)

            # value of the next states with double q learning
            # see https://arxiv.org/abs/1509.06461 for more information on double q learning
            with torch.no_grad():
                next_states = torch.from_numpy(next_states).float().to(self.device)
                idx = row_idx, np.argmax(self.current_critic_net(next_states).cpu().data.numpy(), 1)
                next_state_values = self.target_critic_net(next_states).cpu().data.numpy()[idx]
                next_state_values = next_state_values[not_dones]

            # this defines y = r + discount * max_a q(s', a)
            v_targets = rewards.copy()
            v_targets[not_dones] += self.discount * next_state_values
            v_targets = torch.from_numpy(v_targets).float().to(self.device)

            # this selects only the q values of the actions taken
            idx = row_idx, actions
            states = torch.from_numpy(states).float().to(self.device)
            action_values = self.current_critic_net(states)[idx].float().to(self.device)

            # calculate loss
            # we will take mean later, so calculate raw sum for now
            td_error = self.criterion(v_targets, action_values) * episode_sample_count
            episode_critic_losses.append(td_error)

        # actor gradient step
        self.actor_opt.zero_grad()
        batch_actor_loss = torch.stack(episode_actor_losses).mean()
        batch_actor_loss.backward()
        self.actor_opt.step()

        # critic gradient step
        self.critic_opt.zero_grad()
        batch_critic_loss = torch.stack(episode_critic_losses).sum() / total_sample_count
        batch_critic_loss.backward()
        self.critic_opt.step()

        # reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return batch_actor_loss.clone().item()

    def save(self):
        pass

    def load(self):
        pass
