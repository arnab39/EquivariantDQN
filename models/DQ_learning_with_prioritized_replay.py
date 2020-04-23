import math
import torch
import collections
import sys
print(sys.path)
from memory.prioritized_replay_memory import prioritized_replay_buffer
from tqdm import tqdm
import numpy as np
import torch.autograd as autograd

__all__ = ["DQN_train_prioritized"]

class DQN_train_prioritized():
	def __init__(self, device, network, optimizer, environment, gamma,
			  batch_size, replay_memory_size, epsilon_decay,alpha=0.6,beta=0.4):

		self.device = device
		self.network = network
		self.network.train()
		self.optimizer = optimizer
		self.environment = environment
		self.gamma = gamma
		self.batch_size = batch_size
		self.replay_memory_size = replay_memory_size
		self.alpha = alpha
		self.beta = beta
		self.buffer =  prioritized_replay_buffer(replay_memory_size,alpha)
		self.epsilon_decay = epsilon_decay
		

	def epsilon_by_frame(self, frame_idx):

		epsilon_start = 1.0
		epsilon_final = 0.01
		return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

	def update_network(self):

		USE_CUDA = torch.cuda.is_available()
		Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

		cur_states, actions, next_states, rewards, masks, indices, weights = self.buffer.sample(self.batch_size, self.beta)

		cur_states = Variable(torch.FloatTensor(np.float32(cur_states)))
		next_states = Variable(torch.FloatTensor(np.float32(next_states)))
		actions = Variable(torch.LongTensor(actions))
		rewards = Variable(torch.FloatTensor(rewards))
		masks = Variable(torch.FloatTensor(masks))
		weights = Variable(torch.FloatTensor(weights))

		Q_values = self.network(cur_states)
		next_Q_values = self.network(next_states)

		Q_values = torch.gather(Q_values, 1, actions.unsqueeze(1)).squeeze(1)
		next_Q_values = next_Q_values.max(1)[0]

		delta = rewards + self.gamma * next_Q_values.detach() * masks - Q_values

		loss = delta.pow(2) * weights
		prios = loss + 1e-5
		loss = loss.mean()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.buffer.update_priorities(indices, prios.data.cpu().numpy())

		return loss

	def train_model(self,total_frames, writer):
		
		episode_count = 1
		episode_length = 0
		episode_reward = 0
		cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)
		for frame in tqdm(range(total_frames),desc="Training"):
			epsilon = self.epsilon_by_frame(frame)
			action = self.network.get_action(cur_state, epsilon)
			next_state, reward, done, _ = self.environment.take_action(action)
			next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

			episode_length = episode_length + 1
			episode_reward = episode_reward + reward
			mask = 0 if done else 1
			self.buffer.push(cur_state, action, next_state, reward, mask)

			if len(self.buffer) >= self.batch_size:
				loss = self.update_network()
				writer.add_scalar('Loss', loss, frame)
			cur_state = next_state
			if done:
				writer.add_scalar('Episode length', episode_length, episode_count)
				writer.add_scalar('Reward per episode', episode_reward, episode_count)
				episode_length = 0
				episode_reward = 0
				episode_count = episode_count + 1
				cur_state = torch.tensor(self.environment.reset(), dtype=torch.float32).to(self.device)