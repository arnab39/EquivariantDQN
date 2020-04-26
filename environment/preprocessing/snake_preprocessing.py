import collections
import numpy as np

__all__ = ["frame_history"]

class frame_history():
    def __init__(self, height=32, width =32, frame_history_size=4):
        self.buffer = collections.deque([np.zeros((3, height, width)) for i in range(frame_history_size)])
        self.height = height
        self.width = width
        self.frame_history_size = frame_history_size

    def reset(self,frame):
        self.buffer.clear()
        self.buffer.extend([frame for i in range(self.frame_history_size)])

    def push(self, frame):
        self.buffer.popleft()
        self.buffer.append(frame)

    def get_history(self):
         return np.asarray(self.buffer).reshape((-1, self.height, self.width))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)