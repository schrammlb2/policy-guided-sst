
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

class P2PWrapper(gym.Wrapper)
	def wrapper_setup(self): 
        self.initial_state_vec = copy.deepcopy(self.env.get_state().flatten())
        self.mean = initial_state_vec
        self.std  = np.zeros_like(initial_state_vec)

    def update_norm(self, mean, std):
    	self.mean = mean
    	self.std = std

    def _reset_sim(self):
    	config = self.np_random.uniform(self.mean + self.std/2, self.mean - self.std/2)
        self.env.sim.set_state(config)

        # Randomize start position of object.
        if self.env.has_object:
            object_xpos = self.env.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.env.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.env.initial_gripper_xpos[:2] + self.np_random.uniform(-self.env.obj_range, self.env.obj_range, size=2)
            object_qpos = self.env.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.env.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.env.sim.forward()
        return True