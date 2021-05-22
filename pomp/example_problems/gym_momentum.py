from OpenGL.GL import *
# from .geometric import Set
from ..spaces.objectives import TimeObjectiveFunction, TimeLengthObjectiveFunction
# from ..spaces.controlspace import LambdaKinodynamicSpace, GymWrapperControlSpace
# from ..spaces.configurationspace import MultiConfigurationSpace, BoxConfigurationSpace
from ..spaces.gymwrappers import GymWrapperGoalConditionedControlSpace, RLAgentControlSelector, DDPGAgentWrapper
from ..spaces.gymwrappers import HeuristicWrapper, GDValueSampler
from ..spaces.sets import *#
# from ..spaces.configurationspace import *
# from ..spaces.so2space import SO2Space, so2
from ..spaces.biassets import BoxBiasSet
from ..planners.problem import PlanningProblem

# from ..HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.velocity_env import *
import math
import pickle

"""
Left to implement: 
    Sample from goal set
    Goal set dimension (Must be same as space?)
    contains
    Project
    Signed distance
    Signed distance gradient

Tentative approach
    Treat goal loc as point in relevant dimensions, or neighborhood set
        If goal dim less than state dim, use complex space

    Sample from goal set
        Return goal loc or neighborhood sample, 
        extra dimensions are sampled from state space
    Goal set dimension (Must be same as space?)
        State space dimensions
    contains
        reward == 0
    Project
        goal pos for goal dims, point pos for non-goal-dims
    Signed distance
        distance in goal dims
    Signed distance gradient
        (pos[:goal_dims] - goal)
"""


mean_GD_steps = 10#10
epsilon = 1/(1+mean_GD_steps)

# def pendulum_set_state(self, state):
#     theta = math.asin(state[1])
#     theta_dot = state[-1]
#     # self.env.state = [theta, theta_dot]
#     self.state = [theta, theta_dot]

def set_state(self, state):
    self.pos = np.array(state[:2])
    self.vel = np.array(state[2:])

# class GymPendulum: 
class Momentum: 
    def __init__(self):
        self.env = MultiGoalEnvironment("MultiGoalEnvironment", vel_goal=True, time=True)
        self.env = MultiGoalEnvironment("MultiGoalEnvironment", vel_goal=False, time=True)
        obs = self.env.reset()
        setattr(self.env, 'set_state', set_state)
        # goal = [0,0,-.9,-.9]
        goal = [0,0]
        # goal = self.env.goal.tolist()
        self.control_space = GymWrapperGoalConditionedControlSpace(self.env, goal)

        agent = DDPGAgentWrapper("saved_models/her_MultiGoalEnvironment.pkl", goal_conditioned=True)
        p2p_agent = DDPGAgentWrapper("saved_models/her_MultiGoalEnvironment_p2p.pkl", goal_conditioned=True)
        def make_control_selector(controlSpace,metric,numSamples):
            # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = .5, p_random=.5, goal=goal)
            # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal =1, p_random=.05, goal=goal)
            return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p2p_agent = p2p_agent,
                p_goal = 0, p_random=0, goal=goal)
        self.control_space.controlSelector = make_control_selector


        cs = self.configurationSpace()
        goal_filename = "saved_models/her_MultiGoalEnvironment_value.pkl"
        with open(goal_filename, 'rb') as f:
            goal_value = pickle.load(f)
        p2p_filename = "saved_models/her_MultiGoalEnvironment_value_p2p.pkl"
        with open(p2p_filename, 'rb') as f:
            p2p_value = pickle.load(f)

        start_state = obs['observation'].tolist()
        gd_sampler = GDValueSampler(cs, goal_value, p2p_value, start_state, goal, epsilon=epsilon)
        # gd_sampler = GDValueSampler(cs, goal_value, None, start_state, goal, epsilon=epsilon)
        self.control_space.configuration_sampler = gd_sampler

    def controlSpace(self):
        return self.control_space

    def controlSet(self):
        return self.control_space.action_set

    def startState(self):
        return [-.95, -.95, 0., 0.0]

    def configurationSpace(self):
        return self.control_space.configuration_space
        # res =  MultiConfigurationSpace(SO2Space(),BoxConfigurationSpace([self.omega_min],[self.omega_max]))
        #res.setDistanceWeights([1.0/(2*math.pi),1.0/(self.omega_max-self.omega_min)])
        # return res
    
    def goalSet(self):
        return self.control_space.goal_set

    # def derivative(self,x,u):
    #     theta,omega = x
    #     return [omega,(u[0]/(self.m*self.L**2)-self.g*math.sin(theta)/self.L)]


def gym_momentum_test():
    print("Testing gym pendulum")
    p = Momentum()
    configurationspace = p.configurationSpace()
    assert len(configurationspace.sample()) == 4
    assert configurationspace.contains([0,0,0,0])

    controlset = p.controlSet()
    assert len(controlset.sample()) == 3
    assert controlset.contains([1,0,0])


def gymMomentumTest():
    # gym_momentum_test()
    p = Momentum()
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)
