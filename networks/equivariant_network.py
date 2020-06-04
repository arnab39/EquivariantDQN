import torch
from e2cnn import gspaces
from e2cnn import nn
import random

__all__ = ["D4_steerable_DQN_Snake","D4_steerable_DQN_Snake_new", "steerable_DQN_Pacman"]

class steerable_DQN_Pacman(torch.nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN, rest_lev, modified):
        super(steerable_DQN_Pacman, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.rest_lev = rest_lev
        self.modified = modified
        self.feature_factor = 1
        self.num_feature_fields = [8,16,16,128]
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])
        feature1_type = nn.FieldType(self.r2_act, self.num_feature_fields[0] * [self.r2_act.regular_repr])
        self.feature_field1 = nn.SequentialModule(
            nn.R2Conv(self.input_type, feature1_type, kernel_size=7, padding=2, stride=2, bias=False),
            nn.ReLU(feature1_type, inplace=True),
            nn.PointwiseAvgPoolAntialiased(feature1_type, sigma=0.66, stride=2)
        )

        self.input_feature_type = feature1_type
        if self.rest_lev == 3:
            self.restrict1=self.restrict_layer((0,2),feature1_type)
            self.feature_factor = 2
        else:
            self.restrict1 = lambda x: x
        feature2_type = nn.FieldType(self.r2_act, self.num_feature_fields[1] * self.feature_factor * [self.r2_act.regular_repr])
        self.feature_field2 = nn.SequentialModule(
            nn.R2Conv(self.input_feature_type, feature2_type, kernel_size=5, padding=2, stride=2, bias=False),
            nn.ReLU(feature2_type, inplace=True)
        )

        self.input_feature_type = feature2_type
        if self.rest_lev >=2:
            self.restrict2 = self.restrict_layer((0,1),feature2_type)
            self.feature_factor = 4
        else:
            self.restrict2 = lambda x: x
        feature3_type = nn.FieldType(self.r2_act, self.num_feature_fields[2] * self.feature_factor * [self.r2_act.regular_repr])
        self.feature_field3 = nn.SequentialModule(
            nn.R2Conv(self.input_feature_type, feature3_type, kernel_size=5, padding=1, stride=2, bias=False),
            nn.ReLU(feature3_type, inplace=True)
        )
        if modified:
            self.input_feature_type = feature3_type
            if rest_lev >= 1:
                self.restrict_extra = self.restrict_layer((0,1), feature3_type)
                self.feature_factor = 4
            else:
                self.restrict_extra = lambda x:x
            feature4_type = nn.FieldType(self.r2_act, self.num_feature_fields[3] * self.feature_factor * [self.r2_act.regular_repr])
            self.feature_field_extra = nn.SequentialModule(
                nn.R2Conv(self.input_feature_type, feature4_type, kernel_size=5, padding=0, bias=True),
                nn.ReLU(feature4_type, inplace=True)
            )
            height, width = self.feature_shape()
            self.out_size = self.feature_field_extra.out_type.size
        else:
            self.restrict_extra = lambda  x:x
            self.feature_field_extra = lambda x: x
            height, width = self.feature_shape()
            self.out_size = self.feature_field3.out_type.size * height * width

        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.out_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, self.num_actions)
            )
            self.value = torch.nn.Sequential(  # 2 linear layers
                torch.nn.Linear(self.out_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1)
            )
        else:
            if self.modified:
                self.actionvalue = torch.nn.Linear(self.out_size, self.num_actions)
            else:
                self.actionvalue = torch.nn.Sequential(  # 2 linear layers
                    torch.nn.Linear(self.out_size, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, self.num_actions)
                )

    def restrict_layer(self, subgroup_id, feature_type):
        layers = list()
        layers.append(nn.RestrictionModule(feature_type, subgroup_id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self.input_feature_type = layers[-1].out_type
        self.r2_act = self.input_feature_type.gspace
        return nn.SequentialModule(*layers)

    def feature_shape(self):
        print("Printing network feature fields shape which has restriction level ",self.rest_lev)
        x = torch.zeros(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = nn.GeometricTensor(x, self.input_type)
        x = self.feature_field1(x)
        print(x.shape)
        x = self.feature_field2(self.restrict1(x))
        print(x.shape)
        x = self.feature_field3(self.restrict2(x))
        print(x.shape)
        x = self.feature_field_extra(self.restrict_extra(x))
        print(x.shape)
        x = x.tensor
        b, c, h, w = x.shape
        return h, w

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]) / 255.
        x = nn.GeometricTensor(x, self.input_type)
        x = self.feature_field1(x)
        x = self.feature_field2(self.restrict1(x))
        x = self.feature_field3(self.restrict2(x))
        x = self.feature_field_extra(self.restrict_extra(x))
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





class D4_steerable_DQN_Snake_new(torch.nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN):
        super(D4_steerable_DQN_Snake_new, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])
        feature1_type = nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        feature2_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        feature3_type = nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        feature4_type = nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])

        self.feature_field1 = nn.SequentialModule(
            nn.R2Conv(self.input_type, feature1_type, kernel_size=7, padding=2, stride=2, bias=False),
            nn.ReLU(feature1_type, inplace=True)
        )
        self.feature_field2 = nn.SequentialModule(
            nn.R2Conv(feature1_type, feature2_type, kernel_size=5, padding=1, stride=2, bias=False),
            nn.ReLU(feature2_type, inplace=True)
        )
        self.feature_field3 = nn.SequentialModule(
            nn.R2Conv(feature2_type, feature3_type, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(feature3_type, inplace=True)
        )

        self.equivariant_features = nn.SequentialModule(
            nn.R2Conv(feature3_type, feature4_type, kernel_size=5, stride=1, bias=False),
            nn.ReLU(feature4_type, inplace=True)
        )
        self.gpool = nn.GroupPooling(feature4_type)
        self.feature_shape()
        if self.dueling_DQN:
            print("You are using Dueling DQN")
            self.advantage = torch.nn.Linear(self.equivariant_features.out_type.size, self.num_actions)
            self.value = torch.nn.Linear(self.gpool.out_type.size, 1)
        else:
            self.actionvalue = torch.nn.Linear(self.equivariant_features.out_type.size, self.num_actions)

    def feature_shape(self):
        print("Printing network feature fields shape")
        x = torch.zeros(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = nn.GeometricTensor(x, self.input_type)
        x = self.feature_field1(x)
        print(x.shape)
        x = self.feature_field2(x)
        print(x.shape)
        x = self.feature_field3(x)
        print(x.shape)
        x = self.equivariant_features(x)
        print(x.shape)
        x = x.tensor
        b, c, h, w = x.shape
        return h, w

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])/255.
        x = nn.GeometricTensor(x, self.input_type)
        x = self.feature_field1(x)
        x = self.feature_field2(x)
        x = self.feature_field3(x)
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
