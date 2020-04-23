import collections
import random

__all__ = ["prioritized_replay_buffer"]

########################################
######### To be implemented ############
########################################

class prioritized_replay_buffer(object):
    def __init__(self, memory_size=10000):
        self.buffer = collections.deque()
        self.memory_size = memory_size

    def push(self, transition):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(transition)
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def get(self):
        return zip(*self.buffer)

    def sample(self,batch_size):
        return zip(*random.sample(self.buffer, batch_size))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)