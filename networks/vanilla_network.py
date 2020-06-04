import torch
import torch.nn as nn

import random

__all__ = ["Vanilla_DQN_Snake","DQN_Cartpole", "Vanilla_DQN_Pacman"]

class Vanilla_DQN_Pacman(nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(Vanilla_DQN_Pacman, self).__init__()

        self.input_shape = (input_shape[0], input_shape[1], input_shape[2])
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.features = nn.Sequential( # 3 conv layers
            #nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.Conv2d(self.input_shape[0], 32, kernel_size=7, stride=4, padding=2),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU()
        )
        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions)
            )
            self.value = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.actionvalue = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions)
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]) / 255.
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean()
        else:
            actionvalue = self.actionvalue(x)
            return actionvalue

    def feature_size(self):
        x = self.features(torch.zeros(1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        print(x.size())
        return x.view(1, -1).size(1)

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


class Vanilla_DQN_Snake(nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(Vanilla_DQN_Snake, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.features = nn.Sequential( # 3 conv layers
            nn.Conv2d(self.input_shape[0], 32, kernel_size=7, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU()
        )
        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 256),
                nn.ReLU(),
                nn.Linear(256, self.num_actions)
            )
            self.value = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            self.actionvalue = nn.Sequential(  # 2 linear layers
                nn.Linear(self.feature_size(), 256),
                nn.ReLU(),
                nn.Linear(256, self.num_actions)
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])/255.
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean()
        else:
            actionvalue = self.actionvalue(x)
            return actionvalue

    def feature_size(self):
        x = self.features(torch.zeros(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])).view(1, -1).size(1)
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
