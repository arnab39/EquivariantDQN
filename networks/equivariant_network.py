import torch
#from e2cnn import gspaces
#from e2cnn import nn
import random

__all__ = ["C4_steerable_DQN_Snake","D4_steerable_DQN_Snake", "C4_steerable_DQN_Pacman", "D4_steerable_DQN_Snake"]

class C4_steerable_DQN_Pacman(torch.nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(C4_steerable_DQN_Snake, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])
        feature1_type = nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        feature2_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        feature3_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        self.equivariant_features = nn.SequentialModule(
            nn.R2Conv(self.input_type, feature1_type, kernel_size=7, padding=2, stride=2, bias=False),
            nn.ReLU(feature1_type, inplace=True),
            nn.R2Conv(feature1_type, feature2_type, kernel_size=5, padding=1, stride=2, bias=False),
            nn.ReLU(feature2_type, inplace=True),
            nn.R2Conv(feature2_type, feature3_type, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(feature3_type, inplace=True),
            nn.PointwiseAvgPoolAntialiased(feature3_type, sigma=0.66, stride=1, padding=0)
        )

        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )
            self.value = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            self.actionvalue = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )
    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = nn.GeometricTensor(x, self.input_type)
        x = self.equivariant_features(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean()
        else:
            actionvalue = self.actionvalue(x)
            return actionvalue

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





class C4_steerable_DQN_Snake(torch.nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(C4_steerable_DQN_Snake, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])
        feature1_type = nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        feature2_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        feature3_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        self.equivariant_features = nn.SequentialModule(
            nn.R2Conv(self.input_type, feature1_type, kernel_size=7, padding=2, stride=2, bias=False),
            nn.ReLU(feature1_type, inplace=True),
            nn.R2Conv(feature1_type, feature2_type, kernel_size=5, padding=1, stride=2, bias=False),
            nn.ReLU(feature2_type, inplace=True),
            nn.R2Conv(feature2_type, feature3_type, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(feature3_type, inplace=True),
            nn.PointwiseAvgPoolAntialiased(feature3_type, sigma=0.66, stride=1, padding=0)
        )

        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )
            self.value = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            self.actionvalue = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = nn.GeometricTensor(x, self.input_type)
        x = self.equivariant_features(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean()
        else:
            actionvalue = self.actionvalue(x)
            return actionvalue

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

class D4_steerable_DQN_Snake(torch.nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(D4_steerable_DQN_Snake, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])
        feature1_type = nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        feature2_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        feature3_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])

        self.equivariant_features = nn.SequentialModule(
            nn.R2Conv(self.input_type, feature1_type, kernel_size=7, padding=2, stride=2, bias=False),
            nn.ReLU(feature1_type, inplace=True),
            nn.R2Conv(feature1_type, feature2_type, kernel_size=5, padding=1, stride=2, bias=False),
            nn.ReLU(feature2_type, inplace=True),
            nn.R2Conv(feature2_type, feature3_type, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(feature3_type, inplace=True),
            nn.PointwiseAvgPoolAntialiased(feature3_type, sigma=0.66, stride=1, padding=0)
        )
        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )
            self.gpool = nn.GroupPooling(feature3_type)
            self.value = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.gpool.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            self.actionvalue = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.equivariant_features.out_type.size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions)
            )

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = nn.GeometricTensor(x, self.input_type)
        x = self.equivariant_features(x)
        if self.dueling_DQN:
            x_gpooled = self.gpool(x)
            x_gpooled = x_gpooled.tensor
            x_gpooled = x_gpooled.view(x_gpooled.size(0), -1)
        x = x.tensor
        x = x.view(x.size(0), -1)
        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x_gpooled)
            return value + advantage - advantage.mean()
        else:
            actionvalue = self.actionvalue(x)
            return actionvalue

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
