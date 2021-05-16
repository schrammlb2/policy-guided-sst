# from OpenGL.GL import *
# from .geometric import Set
from ..spaces.objectives import TimeObjectiveFunction, TimeLengthObjectiveFunction
from ..spaces.controlspace import LambdaKinodynamicSpace
from ..spaces.gymwrappers import GymWrapperControlSpace, RLAgentControlSelector, PPOAgentWrapper
from ..spaces.configurationspace import MultiConfigurationSpace, BoxConfigurationSpace
from ..spaces.sets import Set, BoxSet#
# from ..spaces.configurationspace import *
from ..spaces.so2space import SO2Space, so2
from ..spaces.biassets import BoxBiasSet
from ..planners.problem import PlanningProblem
from ..planners.kinodynamicplanner import RandomControlSelector

# from ..spaces.edgechecker import *
# from ..spaces.metric import *
# from ..spaces.interpolators import *
# from ..spaces.geodesicspace import *
# from ..spaces.statespace import *

import math
import gym
import numpy as np
from .gym_pendulum_baseenv import *

class PendulumGoalSet(BoxSet):
    def __init__(self, bmin=-1, bmax=1):
        # self.bmin = [-1 for i in range(3)]
        # self.bmax = [ 1 for i in range(3)]
        v_cap = 2
        # self.bmin = [.9, -1., -v_cap]
        # self.bmax = [1., 1. ,  v_cap]
        self.bmin = [ -1, -1., -v_cap]
        self.bmax = [-.75, 1. ,  v_cap]
        self.box = BoxSet(self.bmin, self.bmax)
    def bounded(self):
        return True


def pendulum_set_state(self, state):
    theta = math.asin(state[1])
    theta_dot = state[-1]
    # self.env.state = [theta, theta_dot]
    self.state = np.array([theta, theta_dot])

# class GymPendulum: 
class Pendulum: 
    def __init__(self):
        env_name = "Pendulum-v0"
        self.env = gym.make(env_name).env
        # self.env = self.env.env
        self.env.reset()
        # setattr(self.env.env, 'set_state', pendulum_set_state)
        # self.control_space = GymWrapperControlSpace(self.env.env)
        setattr(self.env, 'set_state', pendulum_set_state)
        self.control_space = GymWrapperControlSpace(self.env)
        self.control_space.controlSelector = RandomControlSelector#RLAgentControlSelector
        agent = PPOAgentWrapper("saved_models/ppo_" + env_name + ".pkl", goal_conditioned=False)
        def make_control_selector(controlSpace,metric,numSamples):
            # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = 1, p_random=.2)
            # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = 1, p_random=.7, goal_conditioned=False)
            return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = 1, p_random=0, goal_conditioned=False)

        # env_name = "Pendulum_base"
        # self.env = PendulumEnv(g=9.8)
        # self.env.reset()
        setattr(self.env, 'set_state', pendulum_set_state)
        self.control_space = GymWrapperControlSpace(self.env)
        
        self.control_space.controlSelector = make_control_selector
        # etheta = 0.1
        # eomega = 0.5
        # self.goalmin = [math.pi-etheta,-eomega]
        # self.goalmax = [math.pi+etheta,+eomega]

    def controlSpace(self):
        return self.control_space

    def controlSet(self):
        return self.control_space.action_set

    def startState(self):
        return [-1., 0., 0.0]

    def configurationSpace(self):
        return self.control_space.configuration_space
        # res =  MultiConfigurationSpace(SO2Space(),BoxConfigurationSpace([self.omega_min],[self.omega_max]))
        #res.setDistanceWeights([1.0/(2*math.pi),1.0/(self.omega_max-self.omega_min)])
        # return res
    
    def goalSet(self):
        return PendulumGoalSet()

    # def derivative(self,x,u):
    #     theta,omega = x
    #     return [omega,(u[0]/(self.m*self.L**2)-self.g*math.sin(theta)/self.L)]


def gym_pendulum_test():
    print("Testing gym pendulum")
    p = Pendulum()
    configurationspace = p.configurationSpace()
    assert len(configurationspace.sample()) == 3
    assert configurationspace.contains([0,0,0])

    controlset = p.controlSet()
    assert len(controlset.sample()) == 1
    assert controlset.contains([0])


def gymPendulumTest():
    p = Pendulum()
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)
