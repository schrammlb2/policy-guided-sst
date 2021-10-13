import numpy as np
import torch
import random
from collections import deque
# from hyperparams import  OFF_POLICY_BATCH_SIZE as BATCH_SIZE, REPLAY_SIZE, DEVICE
from gym import spaces
import math 

goal_list = []
for i in range(10):
	goal = np.random.rand(2)*2-1
	goal_list.append(goal)
	# self.finished.append(False)

goal_list = [np.zeros(2)]

# class MultiGoalActionSpace:
# 	def __init__(self):
# 		self.shape = (2,)
# 		self.high = (1,)
# 		self.low = (-1,)


# 	def sample(self):
# 		return np.random.rand(2)*2-1




def bounce(pos, vel): 
	for ind in range(pos.shape[0]): 
		if pos[ind] < -1:
			pos[ind] = -1 + abs(-1 - pos[ind])
			vel[ind] = -1*vel[ind]
		elif pos[ind] > 1:
			pos[ind] = 1 - abs(1 - pos[ind])
			vel[ind] = -1*vel[ind]

	return pos, vel

def skate(pos, vel):
	for ind in range(pos.shape[0]): 
		if pos[ind] < -1:
			pos[ind] = -1 + abs(-1 - pos[ind])
			vel[ind] *= 0
		elif pos[ind] > 1:
			pos[ind] = 1 - abs(1 - pos[ind])
			vel[ind] *= 0

	return pos, vel

def brake(pos, vel):
	for ind in range(pos.shape[0]): 
		if pos[ind] < -1:
			pos[ind] = -1 + abs(-1 - pos[ind])
			vel *= 0
		elif pos[ind] > 1:
			pos[ind] = 1 - abs(1 - pos[ind])
			vel *= 0

	return pos, vel


def check_collision(pos):
	for ind in range(pos.shape[0]): 
		if pos[ind] < -1 or pos[ind] > 1:
			return True

	return False

def wall_behavior(pos, vel, abc = (.3, .4, .3)):
	bounce_p, bounce_v = bounce(pos, vel)
	skate_p, skate_v = skate(pos, vel)
	brake_p, brake_v = brake(pos, vel)
	new_pos = abc[0]*bounce_p + abc[1]*skate_p + abc[2]*brake_p
	new_vel = abc[0]*bounce_v + abc[1]*skate_v + abc[2]*brake_v
	return new_pos, new_vel


class MultiGoalEnvironment:
	def __init__(self, name, num=4, time=True, vel_goal=False, epsilon=.1, shift=False, shift_scale=1):
		self.num=1
		# self.pos = np.random.rand(2)*2-1
		# self.vel = np.zeros(2)

		#Linear drag
		scale = 1
		self.max_vel = scale
		self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(2,),
            dtype=np.float32
        )
        
		self.observation_space =  {
			'observation':  spaces.Box(low=-1,high=1,shape=(4,),dtype=np.float32), 
			'achieved_goal':  spaces.Box(low=-1,high=1,shape=(2 + 2*vel_goal,),dtype=np.float32), 
			'desired_goal':  spaces.Box(low=-1,high=1,shape=(2 + 2*vel_goal,),dtype=np.float32)
		}


		self.epsilon=epsilon#.1/4
		# self.step_size = .1/8
		self.step_size = self.epsilon/2
		# self.mass = 20#*self.step_size
		# self.vel_error_margin = 20
		self.mass = 1
		# self.mass = .3
		# self.vel_error_margin = 20
		self.vel_error_margin = .7
		self.v_traction = 2
		# self.action_space = MultiGoalActionSpace()
		self._max_episode_steps = 100
		self.time_limit = self._max_episode_steps
		# self.state_dim = 1+3*num
		self.state_dim = 2
		self.action_dim = 2
		self.time = time
		self.vel_goal = vel_goal
		self.reset()
		# self.buff = deque(maxlen=REPLAY_SIZE)
		self.shift = shift
		self.shift_scale = shift_scale
		

	def reset(self):
		self.pos = np.random.rand(2)*2-1
		self.vel = np.random.rand(2)*2-1#np.zeros(2)
		self.time_limit = self._max_episode_steps
		self.goal = np.random.rand(2)*2-1#np.zeros(2)
		if self.vel_goal:
			self.goal = np.random.rand(2)*2-1
			# rv = np.random.rand(2)*2-1
			#rands = np.random.standard_normal(2)*.3
			#rv = np.clip(rands, -1, 1)
			rv = self.sample_vel_goal()
			self.goal_vel = rv#(self.goal+rv)*.3
			# self.goal_vel = np.zeros(2)#np.clip(self.goal + rv, -1, 1)*
		self.finished = []

		state = self.get_state()
		return state

	def reset_goal(self):
		self.goal = np.random.rand(2)*2-1
		self.goal_vel = self.sample_vel_goal()
		return self.get_state()

	def set_new_vel_goal(self, vel_goal):
		self.goal_vel = vel_goal


	def sample_vel_goal(self):
		rands = np.random.standard_normal(2)*.2
		rv = np.clip(rands, -.8, .8)
		rv = np.random.rand(2)*2-1#
		return rv

	def set_new_goal(self, new_goal):
		self.time_limit = self._max_episode_steps
		self.goal = new_goal
		if self.vel_goal:
			self.goal_vel = np.zeros(2)
		self.finished = []

		state = self.get_state()
		return state

	def set_state(self, pos, vel):
		self.pos = pos
		self.vel = vel

	def get_state(self):
		state = {}
		# state['observation'] = self.pos
		state['observation'] = np.concatenate([self.pos, self.vel], axis=-1)
		if self.vel_goal: 
			state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			state['achieved_goal'] = self.pos
			# state['achieved_goal'] = torch.tensor(self.pos)
			state['desired_goal'] = self.goal

			# state['achieved_goal'] = torch.tensor(np.concatenate([self.pos, self.vel], axis=-1))

		state['done'] = False
		state['reward'] = self.compute_reward(state['achieved_goal'], state['desired_goal'], None)
		state['time'] = self.time_limit
		state['collided'] = False
		return state

	def _get_obs(self): 
		return self.get_state()

	def seed(self, seed):
		np.random.seed(seed)


	def close(self, a, b):
		r_scale = (self.epsilon)**2#*1000000000000000000
		r_dist = ((a[...,:2]-b[...,:2])**2).sum(axis=-1)/r_scale
		dist = r_dist
		if self.vel_goal:
			# v_scale = (self.vel_error_margin/self.mass)**2#*1000000000000
			# v_scale = (self.vel_error_margin/self.mass)**2#*1000000000000
			v_scale = (self.vel_error_margin)**2#*1000000000000
			v_dist = ((a[...,2:]-b[...,2:])**2).sum(axis=-1)/v_scale#*0
			dist = dist + v_dist
			# return (r_dist**.5 < self.epsilon)/2 + (v_dist**.5 < self.epsilon)/2
			# return (r_dist**.5 + v_dist**.5) < self.epsilon
			# return (r_dist**.5 < self.epsilon)*(v_dist**.5 < self.epsilon)

		return dist**.5 < 1#self.epsilon

	# def close(self, a, b):
	# 	r_scale = (self.epsilon)**2#*1000000000000000000
	# 	r_dist = ((a[...,:2]-b[...,:2])**2).sum(axis=-1)/r_scale
	# 	dist = r_dist
	# 	c = dist**.5 < 1
	# 	if self.vel_goal:
	# 		v_scale = (self.vel_error_margin/self.mass)**2#*1000000000000
	# 		v_dist = ((a[...,2:]-b[...,2:])**2).sum(axis=-1)/v_scale#*0
	# 		c = c*(v_dist**.5 < 1)

	# 	return c

	def compute_reward(self, a, b, c):
		# return (((a-b)**2).sum(axis=-1)**.5 < self.epsilon)*1 - (1 if self.time else 0)
		return self.close(a, b)*1 - (1 if self.time else 0)

	def step(self, action, test = False):
		act = np.clip(action, -1, 1).squeeze()
		if self.shift: 
			# act = act - np.array([-.3, -.3])
			act = act - np.array([-.75, -.75])*(self.shift_scale/4)
			force_scale = 1 - .9**(1/self.shift_scale)*np.abs(self.pos)
			# act = act*force_scale*(self.shift_scale/4)
		# force = (act - (self.vel/self.max_vel))
		# force = (act - (self.vel/self.max_vel)*(.1 + .9*((self.vel/self.max_vel)**2).sum()**.5/(2**.5)))
		force = act
		# new_vel = self.vel + force/(self.mass*(1 + np.abs(self.vel).sum()))*self.step_size
		# new_vel = self.vel + force/(self.mass*(1 + self.v_traction*(self.vel**2).sum()))*self.step_size
		# self.vel += np.random.standard_normal()*.00001

		# if self.vel@self.vel < .00001:
		# 	force = act
		# else:
		# 	projection = (act@self.vel)/(self.vel@self.vel)
		# 	mag =  projection*self.vel
		# 	turn =  act - mag
		# 	# mod_turn = turn/(1 + self.v_traction*(self.vel**2).sum()**.5)
		# 	mod_mag = mag/(1 + 2*self.v_traction*(self.vel@self.vel)**.5)
		# 	mod_turn = turn/(1 + self.v_traction*(self.vel@self.vel)**.5)
		# 	force = mod_mag + mod_turn

		new_vel = self.vel + force/self.mass*self.step_size
		# new_vel = self.vel + force/(self.mass*(1 + self.v_traction*(self.vel**2).sum()**.5))*self.step_size
		# new_vel = self.vel + force/(self.mass)*self.step_size
		# new_vel = act
		new_pos = self.pos + new_vel*self.step_size
		# print(next_state['obs'])
		# new_pos = np.clip(new_pos, -1, 1)
		# new_pos, new_vel = bounce(new_pos, new_vel)
		collided = check_collision(new_pos)

		new_pos, new_vel = brake(new_pos, new_vel)
		# new_pos, new_vel = skate(new_pos, new_vel)
		# new_pos, new_vel = wall_behavior(new_pos, new_vel)
		new_vel = np.clip(new_vel, -self.max_vel, self.max_vel)
		self.time_limit -= 1
		if self.close(new_pos, self.goal)*1 > .75:
			reward = 1
			done = True
		else: 
			reward = 0
			done = False

		if self.time_limit <= 0: 
			done = True

		next_state = {}
		next_state['observation'] = np.concatenate([new_pos, new_vel], axis=-1)
		if self.vel_goal: 
			next_state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			next_state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			next_state['achieved_goal'] = self.pos
			next_state['desired_goal'] = self.goal
		next_state['done'] = done
		next_state['time'] = self.time_limit
		next_state['collided'] = collided

		# self.state = next_state

		self.pos = new_pos
		self.vel = new_vel
		info = {}
		info['is_success'] = reward
		if self.time:
			reward -= 1


		# return self.state, reward, done, info
		return next_state, reward, done, info





class RotationGoalEnvironment:
	def __init__(self, name, num=4, time=True, vel_goal=False, epsilon=.1):
		self.num=1
		# self.pos = np.random.rand(2)*2-1
		# self.vel = np.zeros(2)

		#Linear drag
		scale = 1
		self.max_vel = scale
		self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(2,),
            dtype=np.float32
        )
        
		self.observation_space =  spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

		self.epsilon=epsilon#.1/4
		# self.step_size = .1/8
		self.step_size = self.epsilon/2
		self.rotation = 0
		# self.mass = 20#*self.step_size
		# self.vel_error_margin = 20
		self.mass = .5
		# self.vel_error_margin = 20
		self.vel_error_margin = .7
		self.v_traction = 2
		# self.action_space = MultiGoalActionSpace()
		self._max_episode_steps = 100
		self.time_limit = self._max_episode_steps
		# self.state_dim = 1+3*num
		self.state_dim = 2
		self.action_dim = 2
		self.time = time
		self.vel_goal = vel_goal
		self.reset()
		# self.buff = deque(maxlen=REPLAY_SIZE)

	def reset(self):
		self.pos = np.random.rand(2)*2-1
		self.vel = np.random.rand(2)*2-1#np.zeros(2)
		self.time_limit = self._max_episode_steps
		self.goal = np.random.rand(2)*2-1#np.zeros(2)
		if self.vel_goal:
			self.goal = np.random.rand(2)*2-1
			# rv = np.random.rand(2)*2-1
			#rands = np.random.standard_normal(2)*.3
			#rv = np.clip(rands, -1, 1)
			rv = self.sample_vel_goal()
			self.goal_vel = rv#(self.goal+rv)*.3
			# self.goal_vel = np.zeros(2)#np.clip(self.goal + rv, -1, 1)*
		self.finished = []

		state = self.get_state()
		return state

	def reset_goal(self):
		self.goal = np.random.rand(2)*2-1
		self.goal_vel = self.sample_vel_goal()
		return self.get_state()

	def set_new_vel_goal(self, vel_goal):
		self.goal_vel = vel_goal


	def sample_vel_goal(self):
		rands = np.random.standard_normal(2)*.2
		rv = np.clip(rands, -.8, .8)
		rv = np.random.rand(2)*2-1#
		return rv

	def set_new_goal(self, new_goal):
		self.time_limit = self._max_episode_steps
		self.goal = new_goal
		if self.vel_goal:
			self.goal_vel = np.zeros(2)
		self.finished = []

		state = self.get_state()
		return state

	def set_state(self, pos, vel):
		self.pos = pos
		self.vel = vel

	def get_state(self):
		state = {}
		# state['observation'] = self.pos
		state['observation'] = np.concatenate([self.pos, self.vel], axis=-1)
		if self.vel_goal: 
			state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			state['achieved_goal'] = self.pos
			# state['achieved_goal'] = torch.tensor(self.pos)
			state['desired_goal'] = self.goal

			# state['achieved_goal'] = torch.tensor(np.concatenate([self.pos, self.vel], axis=-1))

		state['done'] = False
		state['reward'] = self.compute_reward(state['achieved_goal'], state['desired_goal'], None)
		state['time'] = self.time_limit
		state['collided'] = False
		return state

	def _get_obs(self): 
		return self.get_state()

	def seed(self, seed):
		np.random.seed(seed)


	def close(self, a, b):
		r_scale = (self.epsilon)**2#*1000000000000000000
		r_dist = ((a[...,:2]-b[...,:2])**2).sum(axis=-1)/r_scale
		dist = r_dist
		if self.vel_goal:
			v_scale = (self.vel_error_margin)**2#*1000000000000
			v_dist = ((a[...,2:]-b[...,2:])**2).sum(axis=-1)/v_scale#*0
			dist = dist + v_dist

		return dist**.5 < 1#self.epsilon



	def compute_reward(self, a, b, c):
		# return (((a-b)**2).sum(axis=-1)**.5 < self.epsilon)*1 - (1 if self.time else 0)
		return self.close(a, b)*1 - (1 if self.time else 0)

	def step(self, action, test = False):
		act = np.clip(action, -1, 1).squeeze()
		force = act

		new_vel = self.vel + force/self.mass*self.step_size
		# new_vel = self.vel + force/(self.mass*(1 + self.v_traction*(self.vel**2).sum()**.5))*self.step_size
		# new_vel = self.vel + force/(self.mass)*self.step_size
		new_vel = act
		new_pos = self.pos + new_vel*self.step_size
		# print(next_state['obs'])
		# new_pos = np.clip(new_pos, -1, 1)
		# new_pos, new_vel = bounce(new_pos, new_vel)
		collided = check_collision(new_pos)

		# new_pos, new_vel = brake(new_pos, new_vel)
		# new_pos, new_vel = skate(new_pos, new_vel)
		new_pos, new_vel = wall_behavior(new_pos, new_vel)
		new_vel = np.clip(new_vel, -self.max_vel, self.max_vel)
		self.time_limit -= 1
		if self.close(new_pos, self.goal)*1 > .75:
			reward = 1
			done = True
		else: 
			reward = 0
			done = False

		if self.time_limit <= 0: 
			done = True

		next_state = {}
		next_state['observation'] = np.concatenate([new_pos, new_vel], axis=-1)
		if self.vel_goal: 
			next_state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			next_state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			next_state['achieved_goal'] = self.pos
			next_state['desired_goal'] = self.goal
		next_state['done'] = done
		next_state['time'] = self.time_limit
		next_state['collided'] = collided

		# self.state = next_state

		self.pos = new_pos
		self.vel = new_vel
		info = {}
		info['is_success'] = reward
		if self.time:
			reward -= 1


		# return self.state, reward, done, info
		return next_state, reward, done, info



def square_to_radian(square):
	return ((square + 1)*math.pi)%(2*math.pi)

def radian_to_square(radian):
	return radian%(2*math.pi)/math.pi - 1

class CarEnvironment:
	def __init__(self, name, num=4, time=False, vel_goal=False, epsilon=.1):
		self.num=1
		# self.pos = np.random.rand(2)*2-1
		# self.vel = np.zeros(2)

		#Linear drag
		scale = 1
		self.max_vel = scale


		self.epsilon=epsilon#.1/4
		# self.step_size = .1/8
		self.L = .05
		self.theta_max = 1
		self.step_size = self.epsilon
		self.vel_error_margin = .5#*1000
		# self.action_space = MultiGoalActionSpace()
		self.reward_range = (-1, 0)

		self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(2,),
            dtype=np.float32
        )
        
		self.observation_space =  {'observation': spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        ), 
        'achieved_goal': spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        ), 
        'desired_goal': spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )}

		self._max_episode_steps = 100
		self.time_limit = self._max_episode_steps
		# self.state_dim = 1+3*num
		self.state_dim = 2
		self.action_dim = 2
		self.time = time
		self.vel_goal = vel_goal
		self.vel_goal = False
		self.reset()
		# self.buff = deque(maxlen=REPLAY_SIZE)

	def reset(self):
		self.pos = np.random.rand(2)*2-1
		self.vel = np.random.rand(2)*2-1
		# self.vel = np.random.rand(2)*2*math.pi
		# self.vel[1] = np.random.rand(1)[0]*2-1#np.zeros(2)
		self.time_limit = self._max_episode_steps
		self.goal = np.random.rand(2)*2-1#np.zeros(2)
		if self.vel_goal:
			self.goal = np.random.rand(2)*2-1
			rv = self.sample_vel_goal()
			self.goal_vel = rv
		self.finished = []

		state = self.get_state()
		return state

	def reset_goal(self):
		self.goal = np.random.rand(2)*2-1
		self.goal_vel = self.sample_vel_goal()
		return self.get_state()

	def set_new_vel_goal(self, vel_goal):
		self.goal_vel = vel_goal


	def sample_vel_goal(self):
		rands = np.random.standard_normal(2)*.2
		rv = np.clip(rands, -.8, .8)
		rv = np.random.rand(2)*2-1#
		return rv

	def set_new_goal(self, new_goal):
		self.time_limit = self._max_episode_steps
		self.goal = new_goal
		if self.vel_goal:
			self.goal_vel = np.zeros(2)
		self.finished = []

		state = self.get_state()
		return state

	def set_state(self, pos, vel):
		self.pos = pos
		self.vel = vel

	def get_state(self):
		state = {}
		# state['observation'] = self.pos
		state['observation'] = np.concatenate([self.pos, self.vel], axis=-1)
		if self.vel_goal: 
			state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			state['achieved_goal'] = self.pos
			# state['achieved_goal'] = torch.tensor(self.pos)
			state['desired_goal'] = self.goal

			# state['achieved_goal'] = torch.tensor(np.concatenate([self.pos, self.vel], axis=-1))

		state['done'] = False
		state['reward'] = self.compute_reward(state['achieved_goal'], state['desired_goal'], None)
		state['time'] = self.time_limit
		return state

	def _get_obs(self): 
		return self.get_state()

	def seed(self, seed):
		np.random.seed(seed)


	def close(self, a, b):
		r_scale = (self.epsilon)**2#*1000000000000000000
		r_dist = ((a[...,:2]-b[...,:2])**2).sum(axis=-1)/r_scale
		dist = r_dist
		if self.vel_goal:
			# v_scale = (self.vel_error_margin/self.mass)**2#*1000000000000
			v_scale = (self.vel_error_margin)**2#*1000000000000
			v_dist = ((a[...,2:]-b[...,2:])**2).sum(axis=-1)/v_scale#*0
			dist = dist + v_dist
			# return (r_dist**.5 < self.epsilon)/2 + (v_dist**.5 < self.epsilon)/2
			# return (r_dist**.5 + v_dist**.5) < self.epsilon
			# return (r_dist**.5 < self.epsilon)*(v_dist**.5 < self.epsilon)

		return dist**.5 < 1#self.epsilon

	def compute_reward(self, a, b, c):
		# return (((a-b)**2).sum(axis=-1)**.5 < self.epsilon)*1 - (1 if self.time else 0)
		return self.close(a, b)*1 - (1 if self.time else 0)

	def step(self, action, test = False):
		# if isinstance(action, torch.tensor):
		# action = action.detach().numpy()
		# norm = min(1, (action**2).sum()**.5)
		# norm = max(1, (action**2).sum()**.5)
		# act = (action/norm).squeeze()

		act = np.clip(action, -1, 1).squeeze()
		x_vel = act[0]#(act[0]+1)/2
		turn = act[1]
		d_angle = x_vel/self.L*math.tan(turn*self.theta_max)
		# d_angle = x_vel/.1*math.tan(turn)
		# d_angle = turn
		# new_angle = np.array([(turn+1)*math.pi])
		# angle = turn#self.vel[0]
		# angle = (self.vel[0]+1)*math.pi
		angle = self.vel[0]
		# angle = square_to_radian(self.vel[0])
		# import pdb
		# pdb.set_trace()
		turn_angle = (turn)*math.pi#(angle + self.step_size*d_angle)%(2*math.pi)
		# turn_angle = self.step_size*d_angle
		# new_angle = (angle + turn_angle)%(2*math.pi)
		new_angle = (angle + d_angle)%(2*math.pi)
		# new_angle = (turn_angle)%(2*math.pi)
		# new_angle = angle + turn*math.pi
		# move = x_vel*np.array([math.cos(self.angle[0]), math.sin(self.angle[0])])*self.step_size
		# move = x_vel*np.array([math.cos(new_angle[0]), math.sin(new_angle[0])])*self.step_size
		# move = act*self.step_size
		new_vel = x_vel*np.array([math.cos(new_angle), math.sin(new_angle)])
		move = new_vel*self.step_size
		# move = x_vel*new_angle*self.step_size
		# move = act*self.step_size
		new_pos = np.clip(self.pos+move, -1, 1)

		# new_vel = np.random.rand(2)*2-1
		self.time_limit -= 1
		if self.close(new_pos, self.goal)*1 > .75:
			reward = 1
			done = True
		else: 
			reward = 0
			done = False

		if self.time_limit <= 0: 
			done = True

		next_state = {}
		unit_angle = np.array([radian_to_square(new_angle)])
		# next_state['observation'] = np.concatenate([new_pos, np.array([new_angle])], axis=-1)
		next_state['observation'] = np.concatenate([new_pos, unit_angle], axis=-1)
		# next_state['observation'] = np.concatenate([new_pos, new_vel], axis=-1)
		if self.vel_goal: 
			next_state['achieved_goal'] = np.concatenate([self.pos, self.vel], axis=-1)
			next_state['desired_goal'] = np.concatenate([self.goal, self.goal_vel], axis=-1)
		else:
			next_state['achieved_goal'] = self.pos
			next_state['desired_goal'] = self.goal
		next_state['done'] = done
		# next_state['time'] = self.time_limit

		# self.state = next_state

		self.pos = new_pos
		# self.vel[1] = new_vel[1]
		# self.vel[0] = radian_to_square(new_angle)
		self.vel[0] = new_vel[0]
		info = {}
		info['is_success'] = reward
		if self.time:
			reward -= 1


		return self.get_state(), reward, done, info
		# return next_state, reward, done, info











class Agent:
	def test(self, env=None, verbose = False):
		total_reward = 0
		done = False
		time = 0
		if env == None: 
			# env = StateWrapper("asdf")
			env = MultiGoalEnvironment("asdf", time=True)
			state, done, total_reward = env.reset(), False, 0
		else: 
			state = env.get_state()
		# s1, s2 = state[...,0],state[...,1]
		s1, s2 = state['observation'][...,0],state['observation'][...,1]
		# print("State: [%f, %f] | Total reward: %f " % (s1,s2, total_reward))
		while not done:
			action = self.get_action(state)
			# print(action)
			# print(env._env.pos)
			state, reward, done, _  = env.step(action)
			# total_reward += reward
			time += 1
			s1, s2 = state['observation'][...,0],state['observation'][...,1]
			if verbose:
				print("State: [%f, %f] | Total reward: %f " % (s1,s2, total_reward))
		return -time

class SimpleAgent(Agent):
	def get_action(self, state):
		goal_shape = state['desired_goal'].shape
		action_vec = state['desired_goal'] - state['observation'][:goal_shape[0]]
		action_vec = action_vec/(action_vec**2).sum()**.5
		return action_vec

class NNAgent(Agent):
	def __init__(self, network):
		self.network = network

	def get_action(self, state):
		return self.network.get_actions(state['observation'], state['desired_goal'])


def single_goal_run(initial_pos, goal):
	env = MultiGoalEnvironment("asdf", time=True)
	env.set_state(initial_pos)
	# env.pos = initial_pos[:goal.shape[-1]]
	# env.vel = initial_pos[goal.shape[-1]:]
	env.set_new_goal(goal)
	time = -SimpleAgent().test(env=env)
	return time

def path_runner(initial_pos, path, agent = SimpleAgent()):
	env = MultiGoalEnvironment("asdf", time=True, vel_goal=False)
	env.pos = initial_pos
	time = 0
	for subgoal in path:
		env.set_new_goal(subgoal)
		time -= agent.test(env=env)
	return time



if __name__ == "__main__":
	SimpleAgent().test()

