import collections
import numpy as np
from PIL import Image
import cv2
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt

__all__ = ["frame_history"]

class frame_history():
    def __init__(self, height=84, width =84, frame_history_size=4):
        self.buffer = collections.deque([np.zeros((height, width, 1)) for i in range(frame_history_size)])
        self.height = height
        self.width = width
        self.frame_history_size = frame_history_size

    def reset(self, frame):
        self.buffer.clear()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.buffer.extend([frame[:, :, None] for i in range(self.frame_history_size)])

    def push(self, frame):
        self.buffer.popleft()
        cv2.imshow("first",frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        cv2.imshow("next",frame)
        self.buffer.append(frame[:, :, None])
        #self.buffer.append(frame)

    def get_history(self):
        #print("l="+str(len(self.buffer[0])))
        arr = np.asarray(self.buffer).reshape((-1, self.height, self.width))
        #arr = np.asarray(self.buffer)
        return arr

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)