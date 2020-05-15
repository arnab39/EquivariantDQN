__all__ = ["SnakeEnv_v2"]

from ple.games.snake import Snake
from ple import PLE
import numpy as np
from .preprocessing.snake_preprocessing import frame_history

class SnakeEnv_v2():
    def __init__(self, height=31, width=31, fps=15, frame_history_size=4):  # create the game environment and initialize the attribute values
        self.game = Snake(height, width, init_length=4)
        reward_dict = {"positive": 1.0, "negative": -1.0, "tick": 0.0, "loss": -1.0, "win": 0.0}
        self.environment = PLE(self.game, fps=fps, reward_values=reward_dict, num_steps=2)
        self.init_env()  # initialize the game
        self.input_shape = self.environment.getScreenDims()  # shape of the game input screen
        self.allowed_actions = self.environment.getActionSet()  # the list of allowed actions to be taken by an agent
        self.num_actions = len(self.allowed_actions) - 1  # number of actions that are allowed in this env
        self.frame_hist = frame_history(height=height, width=width, frame_history_size=frame_history_size)
        self.input_shape = self.frame_hist.get_history().shape
        self.map_left = [2, 0, 3, 1]
        self.map_right = [1, 3, 0, 2]

    def init_env(self):
        # initialize the variables and screen of the game
        self.environment.init()

    def get_startaction(self):
        eps = 0.01
        st = self.game.getGameState()
        if st['snake_head_x'] > st['snake_body_pos'][1][0] and abs(
                st['snake_head_y'] - st['snake_body_pos'][1][1]) < eps:
            return 2
        elif st['snake_head_x'] < st['snake_body_pos'][1][0] and abs(
                st['snake_head_y'] - st['snake_body_pos'][1][1]) < eps:
            return 1
        elif st['snake_head_y'] > st['snake_body_pos'][1][1] and abs(
                st['snake_head_x'] - st['snake_body_pos'][1][0]) < eps:
            return 3
        else:
            return 0

    def action_mapper(self, action_snake):
        if action_snake == 0:
            return self.prev_action
        elif action_snake == 1:
            return self.map_left[self.prev_action]
        else:
            return self.map_right[self.prev_action]

    def get_current_state(self):
        # get the current state in the game. Returns the current screen of the game with snake and food positions with
        # a sequence of past .
        cur_frame = np.expand_dims(self.environment.getScreenGrayscale(), axis=0)
        self.frame_hist.push(cur_frame)
        return self.frame_hist.get_history()

    def check_game_over(self):
        # check if the game has terminated
        return self.environment.game_over()

    def reset(self):
        # resets the game to initial values and refreshes the screen with a new small snake and random food position.
        self.environment.reset_game()
        _ = self.environment.act(None)
        self.prev_action = self.get_startaction()
        return self.get_current_state()

    def take_action(self, action):
        # lets the snake take the chosen action of moving in some direction
        action = self.action_mapper(action)
        self.prev_action = action
        action = self.allowed_actions[action]
        reward = self.environment.act(action)
        next_state = self.get_current_state()
        done = self.check_game_over()
        return next_state, reward, done, 0