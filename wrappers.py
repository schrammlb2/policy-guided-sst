
from gym import Wrapper
import numpy as np

class GoalWrapper(Wrapper):
	"""
	Turns a goal-conditioned environment into a regular gym environment
	"""
	def reset(self, **kwargs):
		observation = self.env.reset(**kwargs)
		return self.observation(observation)

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		return self.observation(observation), self.reward(observation), done, info

	def observation(self, obs):
		return np.concatenate([obs['observation'], obs['desired_goal']])

	def reward(self, obs):
		return self.env.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)