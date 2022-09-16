import torch.nn as nn

class VNet(nn.Module):
    # deep v learning architecture
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
        self.linear_state_value = nn.Sequential(nn.Linear(192, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))

    def forward(self, state):
        state = self.convolution_1(state)
        state = self.convolution_2(state)
        state = self.convolution_3(state)
        state = self.convolution_4(state)
        state = state.view(-1, 192)
        state_value = self.linear_state_value(state).reshape(-1, 1)

        return state_value

class QNet(nn.Module):
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
        self.linear_q = nn.Sequential(nn.Linear(192, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, available_actions_count))

    def forward(self, state):
        state = self.convolution_1(state)
        state = self.convolution_2(state)
        state = self.convolution_3(state)
        state = self.convolution_4(state)
        state = state.view(-1, 192)
        q_value = self.linear_q(state)

        return q_value

class DuelQNet(nn.Module):
    # duel dqn architecture
    # see https://arxiv.org/abs/1511.06581 for more information
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
        self.linear_state_value = nn.Sequential(nn.Linear(96, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))
        self.linear_advantage = nn.Sequential(nn.Linear(96, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, available_actions_count))

    def forward(self, state):
        state = self.convolution_1(state)
        state = self.convolution_2(state)
        state = self.convolution_3(state)
        state = self.convolution_4(state)
        state = state.view(-1, 192)

        # subset 1: input for the net to calculate the state value
        # subset 2: relative advantage of actions in the state
        state_subset_1 = state[:, :96]
        state_subset_2 = state[:, 96:]

        state_value = self.linear_state_value(state_subset_1).reshape(-1, 1)
        advantage_values = self.linear_advantage(state_subset_2)
        q_value = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return q_value

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
