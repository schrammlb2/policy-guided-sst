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
from train_distance_function import *

infty = float('inf')
control_duration = 5
constant_extension_chance = 0
# constant_extension_chance = .25

class GymWrapperControlSpace(ControlSpace):
    def __init__(self, env):
        self.env = env
        self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space)
        self.action_set = GymWrapperActionSet(self.env.action_space)
        self.action_dim = self.env.action_space.sample().shape[0]


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
        # for i in range(u[0]):
        #     ghg = self.env.step(np.array(u[1:]))
        assert type(u) == list
        assert type(u[0]) == list
        assert len(u[0]) == self.action_dim
        for control in u:
            ghg = self.env.step(np.array(control))

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
        self.action_dim = self.env.action_space.sample().shape[0]

        def goal_contains(x : list) -> bool: 
            self.env.set_state(self.env, x)
            achieved_goal  = self.env._get_obs()['achieved_goal']
            reward = self.env.compute_reward(achieved_goal, np.array(self.goal), None)
            # reward = self.env.compute_reward(np.array(x[:len(self.goal)]), np.array(self.goal), None)
            return reward > -epsilon
        self.goal_contains = goal_contains

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
        # for i in range(u[0]):
        #     ghg = self.env.step(np.array(u[1:]))
        assert type(u) == list
        assert type(u[0]) == list
        assert len(u[0]) == self.action_dim
        for control in u:
            ghg = self.env.step(np.array(control))

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




# class GDValueSampler(ConfigurationSpace):
#     def __init__(self, configurationSpace, value, start_state, goal, epsilon=.5):
#         self.configurationSpace = configurationSpace
#         self.value = value
#         self.start_state = start_state
#         self.goal = goal
#         self.epsilon = epsilon

#     def sample(self) -> list:
#         k = np.random.geometric(self.epsilon) - 1
#         s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
#         r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
#         # opt = torch.optim.Adam([s], lr=.01)
#         opt = torch.optim.SGD([s], lr=.1)
#         goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
#         start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
#         # if k > 0: 
#         #     import pdb
#         #     pdb.set_trace()
#         for i in range(k):
#             opt.zero_grad()
#             loss = -self.value(s, goal_tensor)
#             loss.backward()
#             opt.step()

#             changed_r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
#             s_projection = start_tensor - (r/changed_r.detach())*(start_tensor - s)
#             s.data = s_projection.data

#         return s.detach().numpy().tolist()

#     def contains(self, x: list) -> bool:
#         return self.configurationSpace.contains(x)

class GDValueSampler(ConfigurationSpace):
    def __init__(self, configurationSpace, goal_value, p2p_value, start_state, goal, epsilon=.5):
        self.configurationSpace = configurationSpace
        self.goal_value = goal_value
        self.p2p_value = p2p_value
        self.start_state = start_state
        self.goal = goal
        self.epsilon = epsilon
        self.total = 0
        self.n = 1

    def sample(self) -> list:
        k = np.random.geometric(self.epsilon) - 1
        s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
        s0 = s.detach()
        r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
        opt = torch.optim.SGD([s], lr=.1)
        # opt = torch.optim.Adam([s], lr=.2)
        goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
        start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
        with torch.no_grad(): 
            # l0 = - self.p2p_value(start_tensor, s)#-self.goal_value(s, goal_tensor) 
            l0 = -self.goal_value(s0, goal_tensor) 

        for i in range(k):
            opt.zero_grad()
            loss = -self.goal_value(s, goal_tensor)# - self.p2p_value(start_tensor, s)
            # loss = (self.goal_value(s, goal_tensor) + self.p2p_value(start_tensor, s))
            loss.backward()
            opt.step()

            # changed_r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
            # s_projection = start_tensor - (r/changed_r.detach())*(start_tensor - s)
            # s.data = s_projection.data

        if k > 10: 
            l1 = -self.goal_value(s, goal_tensor) 
            # l1 = -self.p2p_value(start_tensor, s)
            self.total += (l0 - l1).sum()/k
            self.n += 1
            print(self.total/self.n)

        return s.detach().numpy().tolist()

    # def sample(self) -> list:
    #     k = np.random.geometric(self.epsilon) - 1
    #     s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #     # opt = torch.optim.SGD([s], lr=.1)
    #     opt = torch.optim.Adam([s], lr=.1)
    #     constraint_constant = 30
    #     goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
    #     start_tensor = torch.tensor(self.start_state, dtype=torch.float32)

    #     with torch.no_grad(): 
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         r = g/total

    #     for i in range(k):
    #         opt.zero_grad()
    #         # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         var_r = g/total
    #         loss = -total + constraint_constant*(var_r-r)**2
    #         loss.backward()
    #         opt.step()

    #     return s.detach().numpy().tolist()

    def contains(self, x: list) -> bool:
        return self.configurationSpace.contains(x)



class GDValueSampler(ConfigurationSpace):
    def __init__(self, configurationSpace, goal_value, p2p_value, start_state, goal, epsilon=.5):
        self.configurationSpace = configurationSpace
        self.goal_value = goal_value
        self.p2p_value = p2p_value
        self.start_state = start_state
        self.goal = goal
        self.epsilon = epsilon
        self.pre_total = 0
        self.post_total = 0
        self.n = 1

        from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
        self.env = FetchReachEnv()

    def sample(self) -> list:
        k = np.random.geometric(self.epsilon) - 1
        s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
        s0 = s.detach()
        r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
        opt = torch.optim.SGD([s], lr=.1)
        # opt = torch.optim.Adam([s], lr=.2)
        goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
        start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
        with torch.no_grad(): 
            l0 = -self.goal_value(s, goal_tensor) #- self.p2p_value(start_tensor, s)

        for i in range(k):
            opt.zero_grad()
            # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
            loss = -self.goal_value(s, goal_tensor)#*0
            # loss = (self.goal_value(s, goal_tensor) + self.p2p_value(start_tensor, s))
            loss.backward()
            opt.step()

            # changed_r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
            # s_projection = start_tensor - (r/changed_r.detach())*(start_tensor - s)
            # s.data = s_projection.data


        def state_to_goal(state):
            self.env.sim.set_state_from_flattened(np.array(state.detach()))
            self.env.sim.forward()
            obs = self.env._get_obs()
            return obs['achieved_goal']

        if k > 50: 
            # pregoal = state_to_goal(s0)
            # postgoal = state_to_goal(s)
            # self.pre_total += ((pregoal - self.goal)**2).sum()**.5
            # self.post_total += ((postgoal - self.goal)**2).sum()**.5
            # pregoal = state_to_goal(s0)
            # postgoal = state_to_goal(s)
            l1 = -self.goal_value(s, goal_tensor)
            self.pre_total += l0.detach()
            self.post_total += l1.detach()
            # l1 = -self.goal_value(s, goal_tensor)
            # self.total += (l0 - l1).sum()/k
            self.n += 1
            # print(self.total/self.n)
            print("Pre: " + str(self.pre_total/self.n) + ", Post: " + str(self.post_total/self.n))
            # print(self.total/self.n)

        return s.detach().numpy().tolist()




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

        if type(samp) is np.ndarray: 
            # return_value = duration + samp.tolist()
            return_value = [samp.tolist()]*duration[0]
            return return_value
        else: 
            try: 
                assert type(samp) is int or type(samp) is float
                # return_value = duration +[samp]
                return_value = [samp]*duration
                return return_value 
            except: 
                print("Gym environment does not support list-structured actions")
                # pdb.set_trace()

    def contains(self, x: list) -> bool:
        return True
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
    def __init__(self,controlSpace,metric,numSamples, rl_agent = None, p2p_agent=None, p_goal = .3, p_random=.2, goal = None, goal_conditioned=True):
        self.controlSpace = controlSpace
        self.metric = metric
        self.numSamples = numSamples
        self.rl_agent = rl_agent
        self.p2p_agent = p2p_agent
        self.p_random = p_random
        self.p_goal = p_goal
        self.goal = goal
        self.goal_conditioned = goal_conditioned

        self.constant_extension_chance = constant_extension_chance

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
            return u
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
        duration = len(u)
        if r < self.p_random: 
            return u
        elif r < self.p_random + self.p_goal: 
            #     duration = duration//3 + 1
            state = x
            env = self.controlSpace.env
            env.set_state(env, np.array(x))

            if random.random() <  self.constant_extension_chance: 
                action = self.rl_agent.sample(state, self.goal)
                return [action]*duration
            else:
                sequence = []
                for _ in range(duration): 
                    action = self.rl_agent.sample(state, self.goal)
                    sequence.append(action)
                    obs, reward, done, info = env.step(np.array(action))
                    if self.goal_conditioned: 
                        state = obs['observation']
                    else: 
                        state = obs
                    if reward == 0:
                        break 
            return sequence
            # return [duration] + action
        else: 
            duration = duration//3 + 1
            state = x
            env = self.controlSpace.env
            env.set_state(env, np.array(x))
            if random.random() < self.constant_extension_chance: #.7:
                action = self.p2p_agent.sample(state, xdesired)
                return [action]*duration
            else:
                sequence = []
                for _ in range(duration): 
                    # action = self.rl_agent.sample(state, xdesired)
                    action = self.p2p_agent.sample(state, xdesired)
                    sequence.append(action)
                    obs = env.step(np.array(action))[0]
                    if self.goal_conditioned: 
                        state = obs['observation']
                    else: 
                        state = obs

            return sequence
            # action = self.rl_agent.sample(x, xdesired)
            # return [duration] + action 
            # return [duration] + self.rl_agent.sample(x, xdesired)


    # def _sample_control(self, U, x, xdesired):
    #     r = random.random()
    #     u = U.sample()
    #     duration = len(u)
    #     if r < self.p_random: 
    #         return u
    #     elif r < self.p_random + self.p_goal: 
    #         # if random.random() < .5: 
    #         #     duration = duration//3 + 1
    #         state = x
    #         env = self.controlSpace.env
    #         env.set_state(env, np.array(x))
    #         action = self.rl_agent.sample(state, self.goal)
    #         return [action]*duration

    #         return sequence
            # return [duration] + action
        # else: 
        #     state = x
        #     env = self.controlSpace.env
        #     env.set_state(env, np.array(x))
        #     sequence = []
        #     for _ in range(duration): 
        #         action = self.rl_agent.sample(state, xdesired)
        #         sequence.append(action)
        #         obs = env.step(np.array(action))[0]
        #         if self.goal_conditioned: 
        #             state = obs['observation']
        #         else: 
        #             state = obs

        #     return sequence

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
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0)
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
        try: 
            with open(filename, 'rb') as f:
                self.heuristic = pickle.load(f)
            self.heuristic.eval()
            self.heuristic=self.heuristic.cpu()
        except:
            try: 
                with open(filename, 'rb') as f:
                    self.heuristic = pickle.load(f)
                self.heuristic.eval()
                self.heuristic=self.heuristic.cpu()
            except: 
                print("No p2p time-to-reach function found")
                self.heuristic = None

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