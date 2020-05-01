from argparse import ArgumentParser
from models.DQ_learning import DQN
from networks.vanilla_network import Vanilla_DQN_Snake
from networks.equivariant_network import D4_steerable_DQN_Snake,C4_steerable_DQN_Snake
from environment import SnakeEnv
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import sys

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='EquivariantDQN')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_memory_size', type=int, default=100000)
    parser.add_argument('--epsilon_decay', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=.00001)
    parser.add_argument('--total_episodes', type=int, default=4000)
    parser.add_argument('--summary_dir', type=str, default='runs/c4equivariantnormalreplay_thin')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/c4equivariantnormalreplay_thin')
    parser.add_argument('--gpu_id', type=str, default='0')
    # Available network types: 1) regular 2) C4_equivariant 3) D4_equivariant 4) D4_equivariant_mixedfields
    parser.add_argument('--network_type', type=str, default='D4_equivariant')
    # Select if you want to use priority replay
    parser.add_argument('--priority_replay', type=bool, default=False)
    parser.add_argument('--use_mixedfields', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+args.gpu_id if cuda else "cpu")
    print("You are using ",device," ...")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    environment = SnakeEnv(height= 31, width= 31)
    if args.network_type == 'regular':
        print("You are using Regular CNN...")
        network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
        target_network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
    elif args.network_type == 'C4_equivariant':
        print("You are using C4 Equivariant CNN...")
        network = C4_steerable_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
        target_network = C4_steerable_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
    else:
        print("You are using D4 Equivariant CNN...")
        network = D4_steerable_DQN_Snake(environment.input_shape, environment.num_actions, args.use_mixedfields).to(device)
        target_network = D4_steerable_DQN_Snake(environment.input_shape, environment.num_actions, args.use_mixedfields).to(device)
    if args.priority_replay:
        print("You are using priority replay")
    else:
        print("You are using normal replay")

    print("You are using total ", count_parameters(network), " trainable parameters.")
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    dqn = DQN(device, network, target_network, optimizer, environment, args.gamma,
              args.batch_size, args.replay_memory_size, args.epsilon_decay, args.checkpoint_dir,args.priority_replay)

    if args.train:
        print("You are Training ..")
        writer = SummaryWriter(args.summary_dir)
        dqn.train_model(args.total_episodes, writer)
        writer.close()
    else:
        print("You are Evaluating ..")
        avg_reward = dqn.eval_model(args.total_episodes)
        print("The average reward over ",args.total_episodes," episodes is ",avg_reward)


