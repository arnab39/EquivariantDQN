# This file contains the code that implements the CNN for DQN for the pygame.

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import numpy as np
import random

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

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
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon, environment):
        # function to choose an action
        if random.random() > epsilon:
            # choose from experience (exploitation)
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action_id  = q_value.max(1)[1].data[0]
        else:
            # choose random (exploration)
            action_id = random.randrange(environment.num_actions)
        action = environment.allowed_actions[action_id]
        return action, action_id
