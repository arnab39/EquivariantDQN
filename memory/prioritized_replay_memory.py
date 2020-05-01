import numpy as np

__all__ = ["prioritized_replay_buffer"]


class prioritized_replay_buffer(object):

    def __init__(self, memory_size=100000, prob_alpha=0.6):

        self.prob_alpha = prob_alpha
        self.memory_size = memory_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((memory_size,), dtype=np.float32)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.memory_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.memory_size

    def sample(self, batch_size, beta=0.4):

        if len(self.buffer) == self.memory_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, next_states, rewards, masks = zip(*transitions)
        return states, actions, next_states, rewards, masks, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)