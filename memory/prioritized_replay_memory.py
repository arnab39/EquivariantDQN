import collections
import numpy as np
import random

__all__ = ["prioritized_replay_buffer"]


class prioritized_replay_buffer(object):

	def __init__(self, memory_size=10000, prob_alpha=0.6):

		self.prob_alpha = prob_alpha
		self.memory_size = memory_size
		self.buffer = []
		self.pos = 0
		self.priorities = np.zeros((memory_size,), dtype=np.float32)

	def push(self, state, action, next_state, reward, done):
		
		state = np.expand_dims(state.cuda().data.cpu().numpy(), 0)
		next_state = np.expand_dims(next_state.cuda().data.cpu().numpy(), 0)

		max_prio = self.priorities.max() if self.buffer else 1.0

		if len(self.buffer) < self.memory_size:
		   self.buffer.append((state, action, reward, next_state, done))
		else:
			self.buffer[self.pos] = (state, action, reward, next_state, done)

		self.priorities[self.pos] = max_prio
		self.pos = (self.pos + 1) % self.memory_size

	def sample(self, batch_size, beta=0.4):
		
		if len(self.buffer) == self.memory_size:
			prios = self.priorities
		else:
			prios = self.priorities[:self.pos]

		probs  = prios ** self.prob_alpha
		probs /= probs.sum()

		indices = np.random.choice(len(self.buffer), batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]

		total    = len(self.buffer)
		weights  = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		weights  = np.array(weights, dtype=np.float32)

		batch       = list(zip(*samples))
		states      = np.concatenate(batch[0])
		actions     = batch[1]
		rewards     = batch[2]
		next_states = np.concatenate(batch[3])
		dones       = batch[4]

		return states, actions, next_states, rewards, dones, indices, weights


	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.buffer)