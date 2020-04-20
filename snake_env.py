# This file contains the implementation of the snake game. It uses the PLE library to impleemnt the game.
# It is currently suited for only Snake but can easily be modified for any game. 

from ple.games.snake import Snake
from ple import PLE

import torch


class SnakeEnv():
	def __init__(self, fps, display_screen=True, force_fps=False):
		# create the game envrionment and initialize the attribute values

		self.game = Snake()
		self.environment = PLE(self.game, fps=fps, display_screen=display_screen, force_fps=force_fps)

		self.init_env() # initialize the game 

		self.num_actions = len(self.get_allowed_actions()) # number of actions that are allowed in this env
		self.input_shape = self.get_shape() # shape of the game input screen
		self.allowed_actions = self.environment.getActionSet() # the list of allowed actions to be taken by an agent

	def init_env(self):
		# initialize the variables and screen of the game
		self.environment.init()

	def get_shape(self):
		# get the shape of the game screen
		state = torch.FloatTensor(self.environment.getScreenRGB()).permute(2, 0, 1)
		return list(state.shape)

	def get_allowed_actions(self):
		# get the list of allowed actions
		return self.environment.getActionSet()

	def get_current_state(self):
		# get the current state in the game. Returns the current screen of the game with snake and food positions.
		state = self.environment.getScreenRGB()
		return state

	def check_game_over(self):
		# check if the game has terminated 
		return self.environment.game_over()

	def reset(self):
		# resets the game to initial values and refreshes the screen with a new small snake and random food position.
		self.environment.reset_game()
		return self.get_current_state()

	def take_action(self, action):
		# lets the snake take the chosen action of moving in some direction
		reward = self.environment.act(action)
		next_state = self.get_current_state()
		done = self.check_game_over()
		return next_state, reward, done, 0

