from argparse import ArgumentParser
from models.DQ_learning import DQN
from networks.vanilla_network import Vanilla_DQN_Snake
from networks.vanilla_network import Vanilla_DQN_Pacman
from networks.equivariant_network import D4_steerable_DQN_Snake, D4_steerable_DQN_Snake_new, steerable_DQN_Pacman
from environment import SnakeEnv
from environment import PacmanEnv
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=250000)
    parser.add_argument('--epsilon_decay', type=int, default=40000)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--frame_update_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=.00001)
    parser.add_argument('--total_episodes', type=int, default=3000)
    parser.add_argument('--summary_dir', type=str, default='runs/pacman_equivariant_d4')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pacman_equivariant_d4')
    parser.add_argument('--gpu_id', type=str, default='1')
    # Available network types: 1) regular 2) equivariant
    parser.add_argument('--network_type', type=str, default='equivariant')
    parser.add_argument('--modified', type=bool, default=True)
    parser.add_argument('--restriction_level', type=int, default=0)
    # Available DQN types: 1) regular 2)Dueling
    parser.add_argument('--dueling_DQN', type=bool, default=False)
    # Select if you want to use Double_DQN
    parser.add_argument('--Double_DQN', type=bool, default=True)
    # Select if you want to use priority replay
    parser.add_argument('--priority_replay', type=bool, default=False)
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
    #environment = SnakeEnv(height= 31, width= 31)
    environment = PacmanEnv()
    print(environment.allowed_actions)
    print(environment.num_actions)
    if args.network_type == 'regular':
        print("You are using Regular CNN...")
        #network = Vanilla_DQN_Pacman(environment.input_shape, environment.num_actions, args.dueling_DQN).to(device)
        #target_network = Vanilla_DQN_Pacman(environment.input_shape, environment.num_actions, args.dueling_DQN).to(device)
        network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions, args.dueling_DQN).to(device)
        target_network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions, args.dueling_DQN).to(device)
    else:
        print("You are using Equivariant CNN...")
        network = steerable_DQN_Pacman(environment.input_shape, environment.num_actions, args.dueling_DQN, args.restriction_level, args.modified)
        target_network = steerable_DQN_Pacman(environment.input_shape, environment.num_actions, args.dueling_DQN, args.restriction_level,  args.modified).to(device)
        network = network.to(device)
        #network = D4_steerable_DQN_Snake_new(environment.input_shape, environment.num_actions, args.dueling_DQN)
        #target_network = D4_steerable_DQN_Snake_new(environment.input_shape, environment.num_actions, args.dueling_DQN).to(device)
        #network = network.to(device)
    #sys.exit()
    if args.priority_replay:
        print("You are using priority replay")
    else:
        print("You are using normal replay")

    print("You are using total ", count_parameters(network), " trainable parameters.")
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    dqn = DQN(device, network, target_network, optimizer, environment, args.gamma,
              args.batch_size, args.replay_memory_size, args.epsilon_decay, args.checkpoint_dir, args.Double_DQN, args.epsilon_end, args.priority_replay)

    if args.train:
        print("You are Training ..")
        writer = SummaryWriter(args.summary_dir)
        dqn.train_model(args.total_episodes,args.frame_update_freq, writer)
        writer.close()
    else:
        print("You are Evaluating ..")
        avg_reward = dqn.eval_model(args.total_episodes)
        print("The average reward over ",args.total_episodes," episodes is ",avg_reward)


