# This file contains the implementation of the snake game. It uses the PLE library to implement the game.

__all__ = ["SnakeEnv"]

from ple.games.snake import Snake
from ple import PLE
import numpy as np
from .preprocessing.snake_preprocessing import frame_history

class SnakeEnv():
	def __init__(self, height=32, width=32, fps=15, frame_history_size=4):
		# create the game environment and initialize the attribute values

		self.game = Snake(height=height,width=width,init_length=4)
		reward_dict = {"positive": 1.0, "negative": -1.0, "tick": 0.0, "loss": -1.0, "win": 1.0}
		self.environment = PLE(self.game, fps=fps, reward_values=reward_dict, num_steps=2)
		self.init_env() # initialize the game
		self.allowed_actions = self.environment.getActionSet() # the list of allowed actions to be taken by an agent
		self.num_actions = len(self.allowed_actions) - 1  # number of actions that are allowed in this env
		self.frame_hist = frame_history(height=height,width=width,frame_history_size=frame_history_size,num_channels=3);
		self.input_shape = self.frame_hist.get_history().shape  # shape of the game input screen

	def init_env(self):
		# initialize the variables and screen of the game
		self.environment.init()


	def get_current_state(self):
		# get the current state in the game. Returns the current screen of the game with snake and food positions with
		# a sequence of past .
		cur_frame = np.transpose(self.environment.getScreenRGB(),(2,0,1))
		#cur_frame = np.transpose(np.expand_dims(self.environment.getScreenGrayscale(),axis=0), (2, 0, 1))
		self.frame_hist.push(cur_frame)
		return self.frame_hist.get_history()


	def check_game_over(self):
		# check if the game has terminated
		return self.environment.game_over()

	def reset(self):
		# resets the game to initial values and refreshes the screen with a new small snake and random food position.
		self.environment.reset_game()
		_ = self.environment.act(None)
		self.frame_hist.reset(np.transpose(self.environment.getScreenRGB(),(2,0,1)))
		#self.frame_hist.reset(np.transpose(np.expand_dims(self.environment.getScreenGrayscale(),axis=0), (2, 0, 1)))
		return self.frame_hist.get_history()

	def take_action(self, action):
		# lets the snake take the chosen action of moving in some direction
		reward = self.environment.act(self.allowed_actions[action])
		next_state = self.get_current_state()
		done = self.check_game_over()
		return next_state, reward, done, 0