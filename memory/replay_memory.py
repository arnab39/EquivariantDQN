import collections
import random

__all__ = ["replay_buffer"]

class replay_buffer():
    def __init__(self, memory_size=100):
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