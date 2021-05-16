from .controlspace import ControlSpace
from .configurationspace import ConfigurationSpace
from .sets import Set, LambdaSet
from .interpolators import LinearInterpolator
from ..planners.kinodynamicplanner import ControlSelector


import numpy as np
import random
import pickle
import torch
import gym
from collections import OrderedDict
# from train_distance_function import *

infty = float('inf')
# control_duration = 10
control_duration = 40

class GymWrapperControlSpace(ControlSpace):
    def __init__(self, env):
        self.env = env
        self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space)
        self.action_set = GymWrapperActionSet(self.env.action_space)


        # import pdb
        # pdb.set_trace()
        #

    def configurationSpace(self):
        """Returns the configuration space associated with this."""
        return self.configuration_space

    def controlSet(self,x=None):
        """Returns the set from which controls should be sampled"""
        return self.action_set

    def nextState(self,x,u):
        """Produce the next state after applying control u to state x"""
        self.env.set_state(self.env, np.array(x))
        for i in range(u[0]):
            ghg = self.env.step(np.array(u[1:]))

        return ghg[0].tolist()
        
    def interpolator(self,x,u):
        """Returns the interpolator that goes from x to self.nextState(x,u)"""
        return LinearInterpolator(x, self.nextState(x, u))

    def connection(self,x,y):
        """Returns the sequence of controls that connects x to y, if
        applicable"""
        return None



class GymWrapperGoalConditionedControlSpace(ControlSpace):
    def __init__(self, env, goal: list):
        self.env = env
        # self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space)
        self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space['observation'])
        self.action_set = GymWrapperActionSet(self.env.action_space)
        epsilon = .001
        self.goal = goal

        def goal_contains(x : list) -> bool: 
            self.env.set_state(self.env, x)
            achieved_goal  = self.env._get_obs()['achieved_goal']
            reward = self.env.compute_reward(achieved_goal, np.array(self.goal), None)
            # reward = self.env.compute_reward(np.array(x[:len(self.goal)]), np.array(self.goal), None)
            return reward > -epsilon
        self.goal_set = LambdaSet(goal_contains)#goal may not have right dimensions, fsample=lambda : self.goal)

    def configurationSpace(self):
        """Returns the configuration space associated with this."""
        return self.configuration_space

    def controlSet(self,x=None):
        """Returns the set from which controls should be sampled"""
        return self.action_set

    def nextState(self,x,u):
        """Produce the next state after applying control u to state x"""
        self.env.set_state(self.env, np.array(x))
        for i in range(u[0]):
            ghg = self.env.step(np.array(u[1:]))

        return ghg[0]['observation'].tolist()
        
    def interpolator(self,x,u):
        """Returns the interpolator that goes from x to self.nextState(x,u)"""
        return LinearInterpolator(x, self.nextState(x, u))

    def connection(self,x,y):
        """Returns the sequence of controls that connects x to y, if
        applicable"""
        return None



class GymWrapperConfigurationSpace(ConfigurationSpace):
    def __init__(self, observation_space):
        self.observation_space = observation_space

    def sample(self) -> list:
        samp = self.observation_space.sample()
        # sample_set = [self.observation_space.sample() for i in range(30)]
        # import pdb
        # pdb.set_trace()
        if type(samp) is np.ndarray: 
            return samp.tolist()
        elif type(samp) is int or type(samp) is float: 
            return [samp]
        elif type(samp) == dict or type(samp) == OrderedDict: 
            return samp['observation']
        else:
            import pdb
            pdb.set_trace()
            try: 
                return [samp]
            except: 
                print("Gym environment does not support list-structured observation space")
                # pdb.set_trace()

    def contains(self, x: list) -> bool:
        # import pdb
        # pdb.set_trace()
        assert type(self.observation_space) == gym.spaces.box.Box
        assert self.observation_space.shape[0] == len(x)
        try:
            return self.observation_space.contains(np.array(x))
        except: 
            print("Gym environment does not support checking whether action_space contains value")
            # pdb.set_trace()



class GymWrapperActionSet(Set):
    def __init__(self, action_space):
        self.action_space = action_space
        if hasattr(action_space, 'high') and hasattr(action_space, 'low'):
            self.bounds = lambda : (action_space.low[0], action_space.high[0])
        else: 
            self.bounds = lambda : None

    def sample(self) -> list:
        assert type(self.action_space) == gym.spaces.box.Box
        samp = self.action_space.sample()

        duration = [random.randint(1, control_duration)]
        # duration = [random.randint(1, 25)]
        # duration = [1]

        if type(samp) is np.ndarray: 
            return_value = duration + samp.tolist()
            return return_value
        else: 
            try: 
                assert type(samp) is int or type(samp) is float
                return_value = duration +[samp]
                return return_value 
            except: 
                print("Gym environment does not support list-structured actions")
                # pdb.set_trace()

    def contains(self, x: list) -> bool:
        assert type(self.action_space) == gym.spaces.box.Box
        assert self.action_space.shape[0] == len(x) - 1
        try:
            return self.action_space.contains(np.array(x[1:]))
        except: 
            print("Gym environment does not support checking whether action_space contains value")
            # pdb.set_trace()


class RLAgentControlSelector(ControlSelector):
    """A ControlSelector that randomly samples numSamples controls
    and finds the one that is closest to the destination, according
    to a given metric."""
    def __init__(self,controlSpace,metric,numSamples, rl_agent = None, p_goal = .3, p_random=.2, goal = None):
        self.controlSpace = controlSpace
        self.metric = metric
        self.numSamples = numSamples
        self.rl_agent = rl_agent
        self.p_random = p_random
        self.p_goal = p_goal
        self.goal = goal

    def select(self,x,xdesired):
        ubest = None
        #do we want to eliminate extensions that do not improve the metric
        #from the starting configuration?  experiments suggest no 5/6/2015
        #dbest = self.metric(x,xdesired)
        dbest = infty
        U = self.controlSpace.controlSet(x)#, xdesired)
        if self.numSamples == 1:
            # u = U.sample()
            u = self._sample_control(U, x, xdesired)
            return_value = u #if U.contains(u) else None
            # if self
            # if return_value is None:
            #     return_value = U.sample()
            return return_value
        for iters in range(self.numSamples):
            u = self._sample_control(U, x, xdesired)
            if U.contains(u):
                xnext = self.controlSpace.nextState(x,u)
                d = self.metric(xnext,xdesired)
                if d < dbest:
                    dbest = d
                    ubest = u
        return ubest

    def _sample_control(self, U, x, xdesired):
        r = random.random()
        u = U.sample()
        duration = u[0]
        if r < self.p_random: 
            return u
        elif r < self.p_random + self.p_goal: 
            action = self.rl_agent.sample(x, self.goal)
            # import pdb
            # pdb.set_trace()
            return [duration] + action
        else: 
            action = self.rl_agent.sample(x, xdesired)
            return [duration] + action 
            # return [duration] + self.rl_agent.sample(x, xdesired)


    def _sample_control_sequence(self, U, x, xdesired):
        r = random.random()
        u = U.sample()
        duration = u[0]
        if r < self.p_random: 
            return [u[1:] for _ in duration]
        elif r < self.p_random + self.p_goal: 
            sequence = []
            state = x
            for _ in range(duration):
                action = self.rl_agent.sample(state, self.goal)
            # import pdb
            # pdb.set_trace()
            return [duration] + action
        else: 
            action = self.rl_agent.sample(x, xdesired)
            return [duration] + action 
            # return [duration] + self.rl_agent.sample(x, xdesired)

class RLAgentWrapper:
    def __init__(self, filename, goal_conditioned = True):
        with open(filename, 'rb') as f:
            self.agent = pickle.load(f)

        self.agent.eval()
        self.goal_conditioned = goal_conditioned

    def sample(self, x, goal): 
        raise NotImplementedError

class PPOAgentWrapper(RLAgentWrapper):
    def sample(self, x, goal): 
        x_tensor = torch.tensor(x).unsqueeze(dim=0)
        if self.goal_conditioned: 
            goal_tensor = torch.tensor(goal)
            return_value = self.agent(x_tensor, goal_tensor)[0].sample().detach().squeeze().tolist()
        else:
            return_value =  self.agent(x_tensor)[0].sample().detach().squeeze().tolist()

        if type(return_value) != list:
            return_value = [return_value]

        return return_value


class DDPGAgentWrapper(RLAgentWrapper):
    def sample(self, x, goal): 
        import pdb
        pdb.set_trace()
        x_tensor = torch.tensor(x).unsqueeze(dim=0)
        if self.goal_conditioned: 
            # return_value = normed_forward(self, obs, g, deterministic=False)[0].detach().tolist()
            # return_value = self.agent.normed_forward(x, goal, deterministic=True)[0].detach().tolist()
            return_value = self.agent.normed_forward(x, goal, deterministic=False)[0].detach().tolist()
            # goal_tensor = torch.tensor(goal).unsqueeze(dim=0)
            # inpt = torch.cat((x_tensor, goal_tensor), dim=-1)
            # return_value = self.agent(inpt)[0].detach().tolist()
        else:
            return_value =  self.agent(x_tensor)[0].detach().tolist()

        if type(return_value) != list:
            return_value = [return_value]

        return return_value









class HeuristicWrapper:
    def __init__(self, filename, goal, goal_conditioned = True):
        with open(filename, 'rb') as f:
            self.heuristic = pickle.load(f)

        self.heuristic.eval()
        self.heuristic=self.heuristic.cpu()
        self.goal_conditioned = goal_conditioned
        self.goal = goal

    # def evaluate(self, x): 
    #     x_tensor = torch.tensor(x).unsqueeze(dim=0)
    #     goal_tensor = torch.tensor(self.goal).unsqueeze(dim=0)
    #     return self.heuristic(torch.cat([x_tensor, goal_tensor], dim=-1))
        # raise NotImplementedError

    def evaluate(self, x, goal): 
        x_tensor = torch.tensor(x).unsqueeze(dim=0)
        goal_tensor = torch.tensor(goal).unsqueeze(dim=0)
        return self.heuristic(torch.cat([x_tensor, goal_tensor], dim=-1))