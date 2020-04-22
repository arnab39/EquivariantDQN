import gym

__all__ = ["CartpoleEnv"]

class CartpoleEnv():
    def __init__(self, gamma=0.9, seed=13):
        self.gamma = gamma
        self.seed = seed
        self.environment = gym.make('CartPole-v1')
        self.environment.seed(seed)
        self.num_inputs = self.environment.observation_space.shape[0]
        self.num_actions = self.environment.action_space.n

    def reset(self):
        return self.environment.reset()

    def take_action(self, action):
        return self.environment.step(action)