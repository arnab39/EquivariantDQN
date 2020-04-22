# This file contains the implementation of the snake game. It uses the PLE library to implement the game.
from ple.games.snake import Snake
from ple import PLE

import torch

__all__ = ["SnakeEnv"]

import matplotlib.pyplot as plt
from ple.games.snake import Snake
from ple import PLE
import os
import torch

class SnakeEnv():
	def __init__(self, height=32, width=32, fps=15):
		# create the game environment and initialize the attribute values

		self.game = Snake(height,width,init_length=4)
		reward_dict = {"positive": 2.0, "negative": -1.0, "tick": 0.0, "loss": -5.0, "win": 5.0}
		self.environment = PLE(self.game, fps=fps, reward_values=reward_dict)
		self.init_env() # initialize the game
		self.input_shape = self.environment.getScreenDims() # shape of the game input screen
		self.allowed_actions = self.environment.getActionSet() # the list of allowed actions to be taken by an agent
		self.num_actions = len(self.allowed_actions) - 1  # number of actions that are allowed in this env

	def init_env(self):
		# initialize the variables and screen of the game
		self.environment.init()


	def get_current_state(self):
		# get the current state in the game. Returns the current screen of the game with snake and food positions.
		return self.environment.getScreenRGB()

	def check_game_over(self):
		# check if the game has terminated
		return self.environment.game_over()

	def reset(self):
		# resets the game to initial values and refreshes the screen with a new small snake and random food position.
		self.environment.reset_game()
		_ = self.environment.act(None)
		return self.get_current_state()

	def take_action(self, action):
		# lets the snake take the chosen action of moving in some direction
		action = self.allowed_actions[action]
		reward = self.environment.act(action)
		next_state = self.get_current_state()
		done = self.check_game_over()
		return next_state, reward, done, 0