import torch
import torch.nn as nn
import torch.autograd as autograd

import random

__all__ = ["Vanilla_DQN_Snake","DQN_Cartpole"]

class Vanilla_DQN_Snake(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Vanilla_DQN_Snake, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential( # 3 conv layers
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential( # 2 linear layers
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

    def forward(self, x):
        x = x.view(-1, 3, self.input_shape[0], self.input_shape[1])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        x = self.features(torch.zeros(1, 3, 32, 32)).view(1, -1).size(1)
        return x

    def get_action(self, state, epsilon):
        # function to choose an action
        if random.random() > epsilon:
            # choose from experience (exploitation)
            q_value = self.forward(state)
            action = int(q_value.argmax(-1).detach().data.cpu())
        else:
            # choose random (exploration)
            action = random.randrange(self.num_actions)
        return action

class DQN_Cartpole(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN_Cartpole, self).__init__()
        self.num_actions = num_actions
        self.fc = nn.Sequential(  # 2 linear layers
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, input):
        x = self.fc(input)
        return x

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(state)
            action  = int(q_value.argmax(-1).detach().data.cpu())
        else:
            action = random.randrange(self.num_actions)
        return action