from argparse import ArgumentParser
from models.DQ_learning import DQN_train
#from models.DQ_learning_with_prioritized_replay import DQN_train_prioritized
from networks.vanilla_network import Vanilla_DQN_Snake,DQN_Cartpole
from environment import CartpoleEnv,SnakeEnv
#from plotting import plot_results
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter()

# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='EquivariantDQN')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--replay_memory_size', type=int, default=100000)
    parser.add_argument('--epsilon_decay', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=.00005)
    parser.add_argument('--total_frames', type=int, default=1200000)
    parser.add_argument('--priority', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("You are using ",device," ...")

    # in order to run using catrpole environment, uncomment the following 2 lines and comment the next 3 lines
    #environment = CartpoleEnv(args.gamma,args.seed)
    #network = DQN_Cartpole(environment.num_inputs,environment.num_actions).to(device)

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    environment = SnakeEnv()
    network = Vanilla_DQN_Snake(environment.input_shape, environment.num_actions).to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    #trainer = DQN_train_prioritized(device, network, optimizer, environment, args.gamma,
    #           args.batch_size, args.replay_memory_size, args.epsilon_decay,args.alpha, args.beta)

    trainer = DQN_train(device, network, optimizer, environment, args.gamma,
               args.batch_size, args.replay_memory_size, args.epsilon_decay, args.priority, args.alpha, args.beta)
    trainer.train_model(args.total_frames, writer, args.priority)
    writer.close()


