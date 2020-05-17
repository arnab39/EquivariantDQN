import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

__all__ = ["frame_history"]

class frame_history():
    def __init__(self, height=84, width =84, frame_history_size=4):
        self.buffer = collections.deque([np.zeros((height, width)) for i in range(frame_history_size)])
        self.height = height
        self.width = width
        self.frame_history_size = frame_history_size

    def reset(self, frame):
        self.buffer.clear()
        img = Image.fromarray(frame)
        img = img.resize((self.height, self.width)).convert('L')
        processed_frame = np.array(img)
        self.buffer.extend([processed_frame.astype('uint8') for i in range(self.frame_history_size)])

    def push(self, frame):
        self.buffer.popleft()
        img = Image.fromarray(frame)
        img = img.resize((self.height, self.width)).convert('L')
        processed_frame = np.array(img)
        self.buffer.append(processed_frame.astype('uint8'))

    def get_history(self):
        arr = np.asarray(self.buffer).reshape((-1, self.height, self.width))
        return arr

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
