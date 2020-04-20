import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim

from snake_env import SnakeEnv
from network import CnnDQN
from buffer import ReplayBuffer
from loss import compute_td_loss

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
# function to calculate epsilon decay
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

def plot(frame_idx, rewards, losses):
	# Function to plot the losses and rewards
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

def run(environment, model, optimizer):
	# This function is used for training the model

	replay_initial = 100
	replay_buffer = ReplayBuffer(1000)
	

	num_frames = 14000 # number of steps to run for
	batch_size = 32
	gamma      = 0.99

	losses = []
	all_rewards = []
	episode_reward = 0

	plot_losses = []
	plot_rewards = []

	state = environment.reset()

	for frame_idx in tqdm(range(1, num_frames + 1)):
	    epsilon = epsilon_by_frame(frame_idx) 
	    action, action_id = model.act(state, epsilon, environment)

	    next_state, reward, done, _ = environment.take_action(action)
	    replay_buffer.push(state, action_id, reward, next_state, done)

	    state = next_state
	    episode_reward += reward

	    if done:
	        state = environment.reset()
	        all_rewards.append(episode_reward)
	        episode_reward = 0

	    if len(replay_buffer) > replay_initial:
	        loss = compute_td_loss(model, replay_buffer, optimizer, batch_size)
	        losses.append(loss.data[0])

	    if frame_idx % 1000 == 0:
	        plot_losses.append(np.mean(losses))
	        plot_rewards.append(np.mean(all_rewards))
	        plot(frame_idx, all_rewards, losses)


if __name__ == "__main__":
	# Create game environment object
	os.environ["SDL_VIDEODRIVER"] = "dummy"
	environment = SnakeEnv(fps=0.3) # create the game environment

	model = CnnDQN(environment.input_shape, environment.num_actions) # create the DQN network

	USE_CUDA = torch.cuda.is_available()
	if USE_CUDA:
	    model = model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=0.00001)

	run(environment, model, optimizer)