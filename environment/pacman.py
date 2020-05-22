# This file contains the implementation of the pacman game. It uses the PLE library to implement the game.

__all__ = ["PacmanEnv"]

import gym
import numpy as np
from .preprocessing.pacman_preprocessing import frame_history

class PacmanEnv():
	def __init__(self, height=32, width=32, fps=15, frame_history_size=4):
		# create the game environment and initialize the attribute values
		self.game = 'MsPacmanDeterministic-v4'
		self.environment = gym.make(self.game)
		self.environment.reset() # initialize the game
		self.allowed_actions = self.environment.action_space # the list of allowed actions to be taken by an agent
		self.num_actions = self.environment.action_space.n # number of actions that are allowed in this env
		height = self.environment.observation_space.shape[0]
		width = self.environment.observation_space.shape[1]
		self.frame_hist = frame_history(frame_history_size=frame_history_size)
		self.input_shape = self.frame_hist.get_history().shape  # shape of the game input screen

	def init_env(self):
		# initialize the variables and screen of the game
		self.environment.reset()

	def reset(self):
		# resets the game to initial values and refreshes the screen
		state = self.environment.reset()
		self.frame_hist.reset(state)
		return self.frame_hist.get_history()

	def take_action(self, action):
		# lets the agent take the chosen action of moving in some direction
                next_state, reward, done, _ = self.environment.step(action)
                self.frame_hist.push(next_state)
                next_state = self.frame_hist.get_history()
                reward = np.clip(reward, -1., 1.)
                return next_state, reward, done, _
		

