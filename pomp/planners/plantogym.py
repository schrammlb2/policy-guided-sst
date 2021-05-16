import gym
import numpy as np
from ..spaces.configurationspace import Set, NeighborhoodSubset

class GymSpace:
	def __init__(self, base_space): 
		self.space = base_space
		self.shape = self.sample().shape
		bounds = self.space.bounds()
		self.low = np.array(bounds[0])
		self.high = np.array(bounds[1])

	def sample(self): 
		raise NotImplementedError()

class GymObservationSpace(GymSpace):
	def sample(self): 
		return np.array(self.space.sample())

	def sample_feasible(self): 
		for i in range(1000):
			sample = self.space.sample()
			if self.space.feasible(sample): 
				return sample
		assert False, "Could not find feasible sample"

class GymActionSpace(GymSpace):
	def sample(self): 
		return np.array(self.space.sample()[1:])



class PlanningEnvGymWrapper(gym.Env): 
	def __init__(self, problem, goal_conditioned=True): 
		self.problem = problem
		# self.r = problem.goal.r
		self.observation_space = GymObservationSpace(problem.space.configurationSpace())
		self.action_space = GymActionSpace(problem.space.controlSet(self.problem.start))
		self.goal_conditioned = goal_conditioned
		self._max_episode_steps = 100

	def reset(self, determinisitic=True):
		self.time_remaining = self._max_episode_steps
		if determinisitic: 
			self.state = np.array(self.problem.start)
			self.goal = np.array(self._goal_to_point())
		else: 
			raise NotImplementedError()

		obs = self.state
		goal = self.goal

		if self.goal_conditioned: 
			return {"observation": obs, "achieved_goal": obs, "desired_goal": goal}
		else: 
			return np.concatenate([obs, goal], axis=-1)

	def step(self, action: np.ndarray) -> tuple: 
		assert type(action) == np.ndarray
		if action.shape != ():
			u = action
		elif action.shape == (): 
			u = np.expand_dims(action, axis=0)

		next_state = self._get_next_state(u)
		achieved_goal = next_state
		feasible = self.problem.configurationSpace.feasible(next_state)
		desired_goal = self._goal_to_point()
		reward = self.compute_reward(achieved_goal, desired_goal)
		if self.goal_conditioned: 
			observation = {	"observation": next_state, 
							"achieved_goal": next_state, 
							"desired_goal": np.array(desired_goal), 
							"collided": not feasible}
		else: 
			observation = np.concatenate([next_state, np.array(desired_goal)], axis=-1)
			# observation = next_state

		self.time_remaining -= 1

		done = (reward == 0 or not feasible or (self.time_remaining <= 0))
		info = {'is_success': done and feasible}
		return_value = (observation, reward, done, info)
		return return_value

	def _get_next_state(self, u): 
		return self.problem.controlSpace.nextState(self.state, u)

	def _goal_to_point(self):
		return np.array(self.problem.goal.c)

	def close(self):
		pass

	def compute_reward(self, achieved_goal, desired_goal, info):
		raise NotImplementedError()


class KinomaticGymWrapper(PlanningEnvGymWrapper):
	def __init__(self, problem, dt=.05, goal_conditioned=True): 
		self.problem = problem
		# self.r = problem.goal.r
		self.observation_space = GymObservationSpace(problem.space.configurationSpace())
		self.action_space = GymActionSpace(problem.space.controlSet(self.problem.start))
		self.goal_conditioned = goal_conditioned
		self._max_episode_steps = 50
		self.dt = dt

	def _get_next_state(self, u: np.ndarray) -> np.ndarray: 
		# duration = u[0]
		assert type(u) == np.ndarray
		control = u.tolist()

		# assert self.state in locals(), "State does not exist. Make sure that reset() has been called"

		next_state = self.problem.space.nextState(self.state, [self.dt] + control)
		return np.array(next_state)


	def compute_reward(self, 
			achieved_goal: np.ndarray, 
			desired_goal: np.ndarray, 
			info: dict={}) -> np.ndarray:
		assert type(achieved_goal) == np.ndarray
		assert type(desired_goal) == np.ndarray
		assert achieved_goal.shape == desired_goal.shape
		assert len(achieved_goal.shape) == 1 or len(achieved_goal.shape) == 2
		if len(achieved_goal.shape) == 1: 
			goal_region = self.problem.goal_maker(desired_goal.tolist())
			is_near = goal_region.contains(achieved_goal.tolist())
			if is_near: return np.array(0)
			else: return np.array(-1)
		else: 
			r_list = []
			for a_loc, d_loc in zip(achieved_goal.tolist(), desired_goal.tolist()):
				goal_region = self.problem.goal_maker(d_loc)
				is_near = goal_region.contains(a_loc)
				if is_near: r_list.append(0)
				else: r_list.append(-1)
			return np.array(r_list)


class PendulumWrapper(KinomaticGymWrapper):
	def reset(self, determinisitic=True):
		self.time_remaining = self._max_episode_steps
		# self.state = np.array([2*math.pi, 1])*(np.random.rand(2)*2 -1)

		high = np.array([np.pi, 1])
		# low = np.array([0,-1])
		self.state = np.random.uniform(low=-high, high=high)/100000
		# self.state[0]  = self.state[0]%(2*np.pi)
		self.state = np.array([0,0])
		self.goal = np.array(self._goal_to_point())
		if determinisitic: 
			self.state = np.array(self.problem.start)
			self.goal = np.array(self._goal_to_point())
		# else: 
		# 	raise NotImplementedError()

		obs = self.state
		goal = self.goal

		if self.goal_conditioned: 
			return {"observation": obs, "achieved_goal": obs, "desired_goal": goal}
		else: 
			return np.concatenate([obs, goal], axis=-1)

	def reset(self, determinisitic=True):
		self.time_remaining = self._max_episode_steps
		# self.state = np.array([2*math.pi, 1])*(np.random.rand(2)*2 -1)

		high = np.array([2*np.pi, 10])
		low = np.array([0,-10])
		# self.state = np.random.uniform(low=-high, high=high) 
		self.state =  np.random.uniform(low=low, high=high) 
		self.goal = np.array(self._goal_to_point())
		# if determinisitic: 
		# 	self.state = np.array(self.problem.start)
		# 	self.goal = np.array(self._goal_to_point())
		# else: 
		# 	raise NotImplementedError()

		obs = self.state
		goal = self.goal

		if self.goal_conditioned: 
			return {"observation": obs, "achieved_goal": obs, "desired_goal": goal}
		else: 
			return np.concatenate([obs, goal], axis=-1)

	def compute_reward(self, 
			achieved_goal: np.ndarray, 
			desired_goal: np.ndarray, 
			info: dict={}) -> np.ndarray:
		assert type(achieved_goal) == np.ndarray
		assert type(desired_goal) == np.ndarray
		assert achieved_goal.shape == desired_goal.shape
		assert len(achieved_goal.shape) == 1 or len(achieved_goal.shape) == 2

		return_value = -np.sum((achieved_goal[0] - desired_goal[0])**2, axis=-1) - .1*np.sum((achieved_goal[1] - desired_goal[1])**2, axis=-1) #**.5
		return return_value

	# def compute_reward(self, achieved_goal, desired_goal, info=None):
	# 	neighborhood_set = NeighborhoodSubset(self.problem.space, desired_goal, self.r)
	# 	is_near = neighborhood_set.contains(achieved_goal)
	# 	if is_near: return 0
	# 	else: return -1
		# raise NotImplementedError()