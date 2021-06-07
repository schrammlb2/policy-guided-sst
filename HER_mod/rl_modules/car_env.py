
import gym
from gym import spaces
import math
import numpy as np
import pdb 

def square_to_radian(square):
	return ((square + 1)*math.pi)%(2*math.pi)

def radian_to_square(radian):
	return radian%(2*math.pi)/math.pi - 1

def is_approx(a, b): 
	thresh= .000001
	return abs(a-b) < thresh

class RotationEnv(gym.GoalEnv):
	def __init__(self):
		self.dt = .1
		self.acc_speed = 1
		self.rot_speed = 1
		self.drag = 1

		# observation = [x_pos, y_pos, x_vel, y_vel, rotation]
		# action = [acceleration, turn]

		action_dim = 2
		state_dim = 5
		goal_dim = 2

		self.action_space 	=spaces.Box(-1, 1, shape=(action_dim,), dtype='float32')
		self.observation_space = spaces.Dict(dict(
		    desired_goal	=spaces.Box(-1, 1, shape= (goal_dim,), dtype='float32'),
		    achieved_goal	=spaces.Box(-1, 1, shape= (goal_dim,), dtype='float32'),
		    observation 	=spaces.Box(-1, 1, shape=(state_dim,), dtype='float32'),
		))

		self.threshold = .1


	def reset(self): 
		self.goal  = self.observation_space['desired_goal'].sample()
		state = self.observation_space['observation' ].sample()
		self.position = state[:2]
		self.velocity = state[2:4]
		self.rotation = square_to_radian(state[-1:])
		return self._get_obs()

	def _get_obs(self):
		# observation = np.concatenate([new_position, new_velocity, new_rotation], axis=-1)
		# ag = new_position
		# obs = {'observation': observation, 'achieved_goal': ag, 'desired_goal': self.goal}
		observation = np.concatenate([self.position, self.velocity, radian_to_square(self.rotation)], axis=-1)
		assert type(observation) == np.ndarray
		assert observation.shape == (5,)
		assert self.position.shape == (2,)
		assert self.velocity.shape == (2,)
		assert self.rotation.shape == (1,)
		ag = self.position
		obs = {'observation': observation, 'achieved_goal': ag, 'desired_goal': self.goal}
		return obs

	def _goal_distance(self, a,b): 
		return ((a-b)**2).sum(axis=-1)**.5

	def compute_reward(self, state1, state2, info=None): 
		reward = (self._goal_distance(state1, state2) < self.threshold) - 1
		# reward = -(self._goal_distance(state1, state2))
		return reward

	def step(self, action):
		new_rotation = (self.rotation + action[1]*self.dt*self.rot_speed)%(2*math.pi)
		# new_acceleration = np.array([action[0]*math.cos(self.rotation), action[0]*math.sin(self.rotation)]) - self.drag*self.velocity
		# new_rotation = square_to_radian(action[1:])
		new_acceleration = np.array([action[0]*math.cos(new_rotation), action[0]*math.sin(new_rotation)])# - self.drag*self.velocity
		# new_velocity = np.clip(self.velocity*(1-self.drag) + self.dt*self.acc_speed*new_acceleration, -1, 1)
		# new_velocity = np.clip(self.velocity*(1-self.drag)/(self.dt*self.acc_speed) + self.dt*self.acc_speed*new_acceleration, -1, 1)
		# new_velocity = np.clip(self.velocity*(1-self.drag) + new_acceleration, -1, 1)
		# new_velocity = np.clip(self.velocity + new_acceleration, -1, 1)
		new_velocity = np.clip(self.velocity*(1-self.acc_speed*self.dt) + new_acceleration*self.acc_speed*self.dt, -1, 1)
		# new_velocity = new_acceleration
		new_position = (self.position + new_velocity*self.dt + 1)%2 - 1

		self.rotation = new_rotation
		self.velocity = new_velocity
		self.position = new_position

		obs = self._get_obs()
		reward = self.compute_reward(self.position, self.goal)
		is_success = self._goal_distance(self.position, self.goal) < self.threshold
		info = {'is_success': is_success}

		return obs,reward, False, info

def test(): 
	env = RotationEnv()
	obs = env.reset()
	for _ in range(4):
		print(obs) 
		obs = env.step(env.action_space.sample())


	import pdb
	pdb.set_trace()

# test()