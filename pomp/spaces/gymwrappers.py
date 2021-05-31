from .controlspace import ControlSpace
from .configurationspace import ConfigurationSpace
from .sets import Set, LambdaSet
from .interpolators import LinearInterpolator
from ..planners.kinodynamicplanner import ControlSelector
from ..spaces.plot_gd_sample import plot_gd, approximate_vector_field, scatter_value_heatmap

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
    def __init__(self, env, goal: list, normalize_state_sampling:bool =False):
        self.env = env
        # self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space)
        self.configuration_space = GymWrapperConfigurationSpace(self.env.observation_space['observation'], 
            normalize_state_sampling = normalize_state_sampling)
        self.action_set = GymWrapperActionSet(self.env.action_space)
        epsilon = .001
        self.goal = goal
        self.action_dim = self.env.action_space.sample().shape[0]

        self.normalize_state_sampling = normalize_state_sampling
        if normalize_state_sampling: 
            self.configuration_space.min_s = env._get_obs()['observation']
            self.configuration_space.max_s = env._get_obs()['observation']

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
        x_arr = np.array(x)
        # self.env.set_state(self.env, x_arr)
        self.env.set_state(self.env, x)
        # for i in range(u[0]):
        #     ghg = self.env.step(np.array(u[1:]))
        assert type(u) == list
        assert type(u[0]) == list
        assert len(u[0]) == self.action_dim
        for control in u:
            ghg = self.env.step(np.array(control))

        new_val = ghg[0]['observation']
        if self.normalize_state_sampling: 
            if type(self.configuration_space.min_s) == type(None):
                self.configuration_space.min_s = new_val.copy()
                self.configuration_space.max_s = new_val.copy()
            else:
                self.configuration_space.min_s = np.minimum(self.configuration_space.min_s, new_val)
                self.configuration_space.max_s = np.maximum(self.configuration_space.max_s, new_val)


        return ghg[0]['observation'].tolist()
        
    def interpolator(self,x,u):
        """Returns the interpolator that goes from x to self.nextState(x,u)"""
        return LinearInterpolator(x, self.nextState(x, u))

    def connection(self,x,y):
        """Returns the sequence of controls that connects x to y, if
        applicable"""
        return None



class GymWrapperConfigurationSpace(ConfigurationSpace):
    def __init__(self, observation_space, normalize_state_sampling=False):
        self.observation_space = observation_space
        self.normalize_state_sampling = normalize_state_sampling

        self.min_s = None
        self.max_s = None
        # self.mean_s = None
        # self.var_s = None
        # self.n


    def _sample(self) -> list:
        samp = self.observation_space.sample()

        if type(samp) is np.ndarray: 
            return_samp = samp.tolist()
        elif type(samp) is int or type(samp) is float: 
            return_samp = [samp]
        elif type(samp) == dict or type(samp) == OrderedDict: 
            return_samp = samp['observation']
        else:
            import pdb
            pdb.set_trace()
            try: 
                return_samp = [samp]
            except: 
                print("Gym environment does not support list-structured observation space")
                # pdb.set_trace()

        samp_arr = np.array(return_samp)

        if self.normalize_state_sampling: 
            if type(self.max_s) != type(None):
                diff = self.max_s - self.min_s
                center = self.min_s + diff/2
                new_samp = center + (diff+ .1)*samp_arr 
                # new_samp = center + (diff+.05)*samp_arr/2 
                # if (diff**2).sum()**.5 > 0:#.0000000000000001:
                #     import pdb
                #     pdb.set_trace()
                return new_samp.tolist()
            else: 
                return return_samp
        return return_samp

    def sample(self) -> list:
        self._sample()
        
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
    def __init__(self, configurationSpace, goal_value, p2p_value, start_state, goal, epsilon=.5, zero_buffer=True):
        self.configurationSpace = configurationSpace
        self.goal_value = goal_value
        self.p2p_value = p2p_value
        self.start_state = start_state
        self.goal = goal
        self.epsilon = epsilon
        self.total = 0
        self.n = 1
        self.zero_buffer = zero_buffer
        from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
        self.env = FetchReachEnv()
        self.env.reset()

    # def sample(self) -> list:
    #     k = np.random.geometric(self.epsilon) - 1
    #     s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #     s0 = s.clone().detach()
    #     r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
    #     opt = torch.optim.SGD([s], lr=.1)
    #     # opt = torch.optim.Adam([s], lr=.2)
    #     goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
    #     start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
    #     with torch.no_grad(): 
    #         l0 = - self.p2p_value(start_tensor, s0)-self.goal_value(s0, goal_tensor) 
    #         # l0 = -self.goal_value(s0, goal_tensor) 

    #     for i in range(k):
    #         opt.zero_grad()
    #         # loss = -self.goal_value(s, goal_tensor)# - self.p2p_value(start_tensor, s)
    #         loss = -(self.goal_value(s, goal_tensor) + self.p2p_value(start_tensor, s))
    #         loss.backward()
    #         opt.step()

    #         changed_r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
    #         s_projection = start_tensor - (r/changed_r.detach())*(start_tensor - s)
    #         s.data = s_projection.data

    #     if k > 10: 
    #         # l1 = -self.goal_value(s, goal_tensor) 
    #         # l1 = -self.p2p_value(start_tensor, s)
    #         l1 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor) 
    #         self.total += (l0 - l1).sum()/k
    #         self.n += 1
    #         print(self.total/self.n)

    #     return s.detach().numpy().tolist()

    def sample(self) -> list:
        k = np.random.geometric(self.epsilon) - 1
        if self.zero_buffer:
            s = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
            start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
        else:
            s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
            start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
        # opt = torch.optim.SGD([s], lr=.1)
        s0 = s.detach().clone()
        opt = torch.optim.Adam([s], lr=.05)
        constraint_constant = 30
        goal_tensor = torch.tensor(self.goal, dtype=torch.float32)

        # with torch.no_grad(): 
        #     l0 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor)
        with torch.no_grad(): 
            g = self.goal_value(s, goal_tensor)
            p2p = self.p2p_value(start_tensor, s)
            total = g + p2p
            r = g/total

        traj = [s0]


        def state_to_goal(state):
            assert type(state) == list
            self.env.sim.set_state_from_flattened(np.array([0] + state))
            self.env.sim.forward()
            obs = self.env._get_obs()
            return obs['achieved_goal']

        for i in range(k):
            opt.zero_grad()
            # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
            g = self.goal_value(s, goal_tensor)
            p2p = self.p2p_value(start_tensor, s)
            total = g + p2p
            var_r = g/total
            loss = -total# + constraint_constant*(var_r-r)**2
            # loss = -g
            loss.backward()
            # import pdb
            # pdb.set_trace()
            opt.step()
            traj.append(s.clone().detach())

        # if k > 10: 
        #     # l1 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor) 
        #     # self.total += (l0 - l1).sum()/k
        #     # self.n += 1
        #     # print(self.total/self.n)
        #     from ..spaces.plot_gd_sample import plot_gd
        #     ee_traj = [state_to_goal(t.numpy().tolist()) for t in traj]
        #     ee_start = state_to_goal(self.start_state)
        #     # plot_gd(self.start_state, traj, self.goal)
        #     plot_gd(ee_start, ee_traj, self.goal)
        if self.zero_buffer: 
            rv = s.detach().numpy().tolist()[1:]
        else: 
            rv = s.detach().numpy().tolist()
        # assert len(rv) == 30
        return rv

    # def sample(self) -> list:
    #     k = np.random.geometric(self.epsilon) - 1
    #     if self.zero_buffer:
    #         s = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #         start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
    #     else:
    #         s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #         start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
    #     opt = torch.optim.SGD([s], lr=.000025)
    #     s0 = s.detach().clone()
    #     opt = torch.optim.Adam([s], lr=.01)
    #     constraint_constant = 30
    #     goal_tensor = torch.tensor(self.goal, dtype=torch.float32)

    #     def state_to_goal(state):
    #         # self.env.sim.set_state_from_flattened(np.array(state))
    #         assert type(state) == list
    #         self.env.sim.set_state_from_flattened(np.array([0] + state))
    #         self.env.sim.forward()
    #         obs = self.env._get_obs()
    #         return obs['achieved_goal']


    #     with torch.no_grad(): 
    #         l0 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor)
    #     with torch.no_grad(): 
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         r = g/total

    #     traj = [s0]


    #     for i in range(k):
    #         opt.zero_grad()
    #         # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         var_r = g/total
    #         loss = -total# + constraint_constant*(var_r-r)**2
    #         # loss = -g
    #         loss.backward()
    #         # import pdb
    #         # pdb.set_trace()
    #         opt.step()
    #         traj.append(s.clone().detach())

    #     # if k > 10: 
    #     #     # l1 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor) 
    #     #     # self.total += (l0 - l1).sum()/k
    #     #     # self.n += 1
    #     #     # print(self.total/self.n)
    #     #     # from ..spaces.plot_gd_sample import plot_gd
    #     #     ee_traj = [state_to_goal(t.numpy().tolist()) for t in traj]
    #     #     ee_start = state_to_goal(start_tensor.detach().numpy().tolist())
    #     #     plot_gd(ee_start, ee_traj, self.goal)
    #         # pass
    #     if self.zero_buffer: 
    #         rv = s.detach().numpy().tolist()[1:]
    #     else: 
    #         rv = s.detach().numpy().tolist()
    #     # assert len(rv) == 30
    #     return rv




    # def sample(self) -> list:
          #
    #     #Plot gradient at some sampled points
          #
    #     k = np.random.geometric(self.epsilon) - 1
    #     # opt = torch.optim.Adam([s], lr=.05)
    #     constraint_constant = 30
    #     goal_tensor = torch.tensor(self.goal, dtype=torch.float32)


    #     grad_list = []


    #     def state_to_goal(state):
    #         # self.env.sim.set_state_from_flattened(np.array(state))
    #         assert type(state) == list
    #         self.env.sim.set_state_from_flattened(np.array([0] + state))
    #         self.env.sim.forward()
    #         obs = self.env._get_obs()
    #         return obs['achieved_goal']

    #     # def state_to_goal(state):
    #     #     self.env.sim.set_state_from_flattened(np.array(state))
    #     #     self.env.sim.forward()
    #     #     obs = self.env._get_obs()
    #     #     return obs['achieved_goal']

    #     # def state_to_goal(state):
    #     #     return state[:2]


    #     for i in range(100):
    #         if self.zero_buffer:
    #             s = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #             start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
    #         else:
    #             s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #             start_tensor = torch.tensor(self.start_state, dtype=torch.float32)

    #         # opt = torch.optim.SGD([s], lr=.005)
    #         opt = torch.optim.SGD([s], lr=.000025)
    #         s0 = s.detach().clone()

    #         opt.zero_grad()
    #         # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         var_r = g/total
    #         # loss = -total 
    #         loss = -g
    #         loss.backward()
    #         opt.step()
    #         grad_list.append([s0, s.clone().detach()])
    #         # traj.append()

    #     # if k > 10: 
    #     #     # l1 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor) 
    #     #     # self.total += (l0 - l1).sum()/k
    #     #     # self.n += 1
    #     #     # print(self.total/self.n)
    #     #     from ..spaces.plot_gd_sample import plot_gd
    #     # import pdb
    #     # pdb.set_trace()
    #     ee_traj = [ [state_to_goal(t[0].numpy().tolist()),  state_to_goal(t[1].numpy().tolist())]
    #         for t in grad_list]
    #     ee_start = state_to_goal(start_tensor.detach().numpy().tolist())
    #     #     # plot_gd(self.start_state, traj, self.goal)
    #     approximate_vector_field(ee_start, ee_traj, self.goal)

    #     if self.zero_buffer: 
    #         rv = s.detach().numpy().tolist()[1:]
    #     else: 
    #         rv = s.detach().numpy().tolist()
    #     # assert len(rv) == 30
    #     return rv


    # def sample(self) -> list:
    #     #
    #     #Plot value of some sampled points
    #     #
    #     k = np.random.geometric(self.epsilon) - 1
    #     # opt = torch.optim.Adam([s], lr=.05)
    #     constraint_constant = 30
    #     goal_tensor = torch.tensor(self.goal, dtype=torch.float32)

    #     val_list = []


    #     def state_to_goal(state):
    #         # self.env.sim.set_state_from_flattened(np.array(state))
    #         assert type(state) == list
    #         self.env.sim.set_state_from_flattened(np.array([0] + state))
    #         self.env.sim.forward()
    #         obs = self.env._get_obs()
    #         return obs['achieved_goal']
            
    #     # def state_to_goal(state):
    #     #     self.env.sim.set_state_from_flattened(np.array(state))
    #     #     self.env.sim.forward()
    #     #     obs = self.env._get_obs()
    #     #     return obs['achieved_goal']

    #     # def state_to_goal(state):
    #     #     return state[:2]


    #     for i in range(1000):
    #         if self.zero_buffer:
    #             s = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #             start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
    #         else:
    #             s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
    #             start_tensor = torch.tensor(self.start_state, dtype=torch.float32)

    #         s0 = s.detach().clone()

    #         # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
    #         g = self.goal_value(s, goal_tensor)
    #         p2p = self.p2p_value(start_tensor, s)
    #         total = g + p2p
    #         var_r = g/total
    #         # loss = -total 
    #         loss = -g
    #         achieved_state = state_to_goal(s.detach().numpy().tolist())
    #         val_list.append([achieved_state, loss.detach().numpy().tolist()])
    #         # traj.append()

    #     ee_traj = [(t[0][0], t[0][1], t[1]) for t in val_list]
    #     ee_start = state_to_goal(start_tensor.detach().numpy().tolist())
    #     #     # plot_gd(self.start_state, traj, self.goal)
    #     scatter_value_heatmap(ee_start, ee_traj, self.goal)

    #     if self.zero_buffer: 
    #         rv = s.detach().numpy().tolist()[1:]
    #     else: 
    #         rv = s.detach().numpy().tolist()
    #     # assert len(rv) == 30
    #     return rv

    def contains(self, x: list) -> bool:
        return self.configurationSpace.contains(x)



# class GDValueSampler(ConfigurationSpace):
#     def __init__(self, configurationSpace, goal_value, p2p_value, start_state, goal, epsilon=.5):
#         self.configurationSpace = configurationSpace
#         self.goal_value = goal_value
#         self.p2p_value = p2p_value
#         self.start_state = start_state
#         self.goal = goal
#         self.epsilon = epsilon
#         self.pre_total = 0
#         self.post_total = 0
#         self.n = 1

#         from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
#         self.env = FetchReachEnv()

#     def sample(self) -> list:
#         k = np.random.geometric(self.epsilon) - 1
#         s = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32, requires_grad=True)
#         s0 = s.detach().clone()
#         # r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
#         r = ((torch.tensor(self.start_state[:2]) - s[:2])**2).sum()**.5
#         # opt = torch.optim.SGD([s], lr=.1)
#         opt = torch.optim.Adam([s], lr=.05)
#         goal_tensor = torch.tensor(self.goal, dtype=torch.float32)
#         start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
#         with torch.no_grad(): 
#             l0 = -self.goal_value(s, goal_tensor) #- self.p2p_value(start_tensor, s)


#         # def state_to_goal(state):
#         #     self.env.sim.set_state_from_flattened(np.array(state.detach()))
#         #     self.env.sim.forward()
#         #     obs = self.env._get_obs()
#         #     return obs['achieved_goal']

#         def state_to_goal(state):
#             return state[:2]

#         traj = [s0]

#         for i in range(k):
#             # achieved_goal = state_to_goal(s)
#             # if k > 50 and self.n > 10: 
#             #     import pdb
#             #     pdb.set_trace()
#             opt.zero_grad()
#             # loss = -self.goal_value(s, goal_tensor) - self.p2p_value(start_tensor, s)
#             # loss = -self.goal_value(s, goal_tensor)#*0
#             loss = -(self.goal_value(s, goal_tensor) + self.p2p_value(start_tensor, s))
#             loss.backward()
#             opt.step()
#             # traj.append(s.clone().detach())

#             # changed_r = ((torch.tensor(self.start_state) - s)**2).sum()**.5
#             # s_projection = start_tensor - (r/changed_r.detach())*(start_tensor - s)
#             # s.data = s_projection.data


#             # changed_r = ((torch.tensor(self.start_state[:2]) - s[:2])**2).sum()**.5
#             # s_projection = start_tensor[:2] - (r/changed_r.detach())*(start_tensor[:2] - s[:2])
#             # s.data[:2] = s_projection.data#[:2]
#             traj.append(s.clone().detach())



#         if k > 10: 
#             pregoal = state_to_goal(s0)
#             postgoal = state_to_goal(s)
#             self.pre_total += ((pregoal - torch.tensor(self.goal))**2).sum()**.5
#             self.post_total += ((postgoal - torch.tensor(self.goal))**2).sum()**.5
#         #     l1 = -self.goal_value(s, goal_tensor)
#         #     self.pre_total += l0.detach()
#         #     self.post_total += l1.detach()
#         #     # l1 = -self.goal_value(s, goal_tensor)
#         #     # self.total += (l0 - l1).sum()/k
#             self.n += 1
#         #     # print(self.total/self.n)
#             print("Pre: " + str(self.pre_total/self.n) + ", Post: " + str(self.post_total/self.n))
#             from ..spaces.plot_gd_sample import plot_gd
#             plot_gd(self.start_state, traj, self.goal)
#         #     # print(self.total/self.n)

#         return s.detach().numpy().tolist()



class GDValueSampler(ConfigurationSpace):
    def __init__(self, configurationSpace, goal_value, p2p_value, start_state, goal, 
        norm=None, denorm=None, epsilon=.5, zero_buffer=True):
        self.configurationSpace = configurationSpace
        self.goal_value = goal_value
        self.p2p_value = p2p_value
        self.start_state = start_state
        self.goal = goal
        self.epsilon = epsilon
        self.total = 0
        self.n = 1
        self.zero_buffer = zero_buffer
        from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
        self.env = FetchReachEnv()
        self.env.reset()

        if type(norm) == type(None):
            self.norm = lambda x, y: (x, y)
            assert False
        else: 
            self.norm = norm
        if type(denorm) == type(None):
            self.denorm = lambda x, y: (x, y)
            assert False
        else: 
            self.denorm = denorm


    def sample(self) -> list:
        k = np.random.geometric(self.epsilon) - 1

        # sample = self.configurationSpace.sample()
        # sample_norm, g_norm = self.norm(torch.tensor(sample, dtype=torch.float32), 
        #                                 torch.tensor(self.goal, dtype=torch.float32))

        #configuration space sampler is standard gaussian
        if self.zero_buffer:
            # sample_norm = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32)
            sample_norm, g_norm = self.norm(torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32), 
                                        torch.tensor(self.goal, dtype=torch.float32))
            # sample_norm = torch.tensor([0] + np.random.randn(self.shape), dtype=torch.float32)
            start_norm, g_norm = self.norm(torch.tensor([0] + self.start_state, dtype=torch.float32), 
                                        torch.tensor(self.goal, dtype=torch.float32))
            start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
        else:
            # sample_norm = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32)
            # sample_norm = torch.tensor(np.random.randn(self.shape), dtype=torch.float32)
            sample_norm, g_norm = self.norm(torch.tensor(self.configurationSpace.sample(), dtype=torch.float32), 
                                        torch.tensor(self.goal, dtype=torch.float32))
            start_norm, g_norm = self.norm(   torch.tensor(self.start_state, dtype=torch.float32), 
                                        torch.tensor(self.goal, dtype=torch.float32))
            start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
        # if self.zero_buffer: 
        #     sample_norm = torch.tensor([0] + self.configurationSpace.sample(), dtype=torch.float32)
        #     start_norm, _ = self.norm(torch.tensor([0] + self.start_state, dtype=torch.float32), 
        #                                 torch.tensor(self.goal, dtype=torch.float32))
        #     start_tensor = torch.tensor([0] + self.start_state, dtype=torch.float32)
        # else: 
        #     sample_norm = torch.tensor(self.configurationSpace.sample(), dtype=torch.float32)
        #     start_norm, _ = self.norm(   torch.tensor(self.start_state, dtype=torch.float32), 
        #                                 torch.tensor(self.goal, dtype=torch.float32))
        #     start_tensor = torch.tensor(self.start_state, dtype=torch.float32)
        # _ , g_norm = self.norm(sample_norm, torch.tensor(self.goal, dtype=torch.float32))

        s0 = sample_norm.detach().clone()
        s_norm = sample_norm.detach().requires_grad_()
        opt = torch.optim.Adam([s_norm], lr=.05)

        constraint_constant = 30

        # with torch.no_grad(): 
        #     l0 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor)
        with torch.no_grad(): 
            # g = self.goal_value(s, goal_tensor)
            # p2p = self.p2p_value(start_tensor, s)
            g = self.goal_value(s_norm, g_norm, norm=False)
            p2p = self.p2p_value(start_norm, s_norm, norm=False)
            total = g + p2p
            r = g/total

        traj = [s0]


        def state_to_goal(state):
            assert type(state) == list
            self.env.sim.set_state_from_flattened(np.array([0] + state))
            self.env.sim.forward()
            obs = self.env._get_obs()
            return obs['achieved_goal']

        for i in range(k):
            opt.zero_grad()
            # g = self.goal_value(s, goal_tensor)
            # p2p = self.p2p_value(start_tensor, s)
            g = self.goal_value(s_norm, g_norm, norm=False)
            s, _ = self.denorm(s_norm, g_norm)
            p2p = self.p2p_value(start_tensor, s)
            # p2p = self.p2p_value(start_norm, s_norm, norm=False)
            total = g + p2p
            var_r = g/total
            reg_loss = 1*(s_norm**2).sum()
            loss = -total + constraint_constant*(var_r-r)**2
            # loss = -total #+ reg_loss
            # loss = -g
            loss.backward()
            # import pdb
            # pdb.set_trace()
            opt.step()
            traj.append(s_norm.clone().detach())

        # if k > 10: 
        #     # l1 = - self.p2p_value(start_tensor, s)-self.goal_value(s, goal_tensor) 
        #     # self.total += (l0 - l1).sum()/k
        #     # self.n += 1
        #     # print(self.total/self.n)
        #     from ..spaces.plot_gd_sample import plot_gd
        #     ee_traj = [state_to_goal(t.numpy().tolist()) for t in traj]
        #     ee_start = state_to_goal(self.start_state)
        #     # plot_gd(self.start_state, traj, self.goal)
        #     plot_gd(ee_start, ee_traj, self.goal)

        s, _ = self.denorm(s_norm, g_norm)

        if self.zero_buffer: 
            rv = s.detach().numpy().tolist()[1:]
        else: 
            rv = s.detach().numpy().tolist()
        # assert len(rv) == 30
        return rv


    def contains(self, x: list) -> bool:
        return self.configurationSpace.contains(x)




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
        assert type(x) == list
        if r < self.p_random: 
            return u
        elif r < self.p_random + self.p_goal: 
            #     duration = duration//3 + 1
            state = x
            env = self.controlSpace.env
            # env.set_state(env, np.array(x))
            env.set_state(env, x)

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
                        state = obs['observation'].tolist()
                    else: 
                        state = obs.tolist()
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
                        state = obs['observation'].tolist()
                    else: 
                        state = obs.tolist()

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
    def __init__(self, filename, goal_conditioned = True, zero_buffer=True):
        print('Opening policy at location ' + filename)
        with open(filename, 'rb') as f:
            self.agent = pickle.load(f)

        self.agent.eval()
        self.goal_conditioned = goal_conditioned
        self.zero_buffer = zero_buffer

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
    def sample(self, x:list, goal:list) -> list: 
        # assert type(x) == np.ndarray
        assert type(x) == list
        assert type(goal) == list
        if self.goal_conditioned: 
            if self.zero_buffer:
                obs = [0] + x
            else:
                obs = x

            return_value = self.agent.normed_forward(np.array(obs), np.array(goal), deterministic=False)[0].detach().tolist()
            # return_value = normed_forward(self, obs, g, deterministic=False)[0].detach().tolist()
            # return_value = self.agent.normed_forward(x, goal, deterministic=True)[0].detach().tolist()
            # goal_tensor = torch.tensor(goal).unsqueeze(dim=0)
            # inpt = torch.cat((x_tensor, goal_tensor), dim=-1)
            # return_value = self.agent(inpt)[0].detach().tolist()
        else:
            x_tensor = torch.tensor(x).unsqueeze(dim=0)
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