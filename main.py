from argparse import ArgumentParser
from models.DQ_learning import DQN_train
from networks.vanilla_network import Vanilla_DQN_Snake,DQN_Cartpole
from networks.equivariant_network import D4_steerable_DQN_Snake,C4_steerable_DQN_Snake
from environment import CartpoleEnv,SnakeEnv
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
import os


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='EquivariantDQN')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_memory_size', type=int, default=100000)
    parser.add_argument('--epsilon_decay', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=.00001)
    parser.add_argument('--total_frames', type=int, default=2000000)
    parser.add_argument('--summary_dir', type=str, default='runs/d4equivariantcnn_thick_numsteps2')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/d4equivariantcnn_thick_numsteps2')
    parser.add_argument('--gpu_id', type=str, default='0')
    # Available network types: 1) regular 2) C4_equivariant 3) D4_equivariant
    parser.add_argument('--network_type', type=str, default='D4_equivariant')
    # Select if you want to use priority replay
    parser.add_argument('--priority_replay', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+args.gpu_id if cuda else "cpu")
    print("You are using ",device," ...")

    writer = SummaryWriter(args.summary_dir)

    #environment = CartpoleEnv(args.gamma,args.seed)
    #network = DQN_Cartpole(environment.num_inputs,environment.num_actions).to(device)

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    environment = SnakeEnv(height=31, width=31)
    if args.network_type == 'regular':
        print("You are using Regular CNN...")
        network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
    elif args.network_type == 'C4_equivariant':
        print("You are using C4 Equivariant CNN...")
        network = C4_steerable_DQN_Snake(environment.input_shape, environment.num_actions).to(device)
    else:
        print("You are using D4 Equivariant CNN...")
        network = D4_steerable_DQN_Snake(environment.input_shape, environment.num_actions).to(device)

    if args.priority_replay:
        print("You are using priority replay")
    else:
        print("You are using normal replay")

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    trainer = DQN_train(device, network, optimizer, environment, args.gamma,
              args.batch_size, args.replay_memory_size, args.epsilon_decay, args.checkpoint_dir,args.priority_replay)

    trainer.train_model(args.total_frames, writer)

    writer.close()


