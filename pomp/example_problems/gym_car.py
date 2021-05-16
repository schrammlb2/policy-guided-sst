from OpenGL.GL import *
# from .geometric import Set
from ..spaces.objectives import TimeObjectiveFunction
# from ..spaces.controlspace import LambdaKinodynamicSpace, GymWrapperControlSpace
# from ..spaces.configurationspace import MultiConfigurationSpace, BoxConfigurationSpace
from ..spaces.gymwrappers import GymWrapperGoalConditionedControlSpace, RLAgentControlSelector, DDPGAgentWrapper
from ..spaces.sets import *#
# from ..spaces.configurationspace import *
# from ..spaces.so2space import SO2Space, so2
from ..spaces.biassets import BoxBiasSet
from ..planners.problem import PlanningProblem

# from ..HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.velocity_env import CarEnvironment
import math

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


use_agent = True
# p_goal = .5
p_random = .5
p_goal = 1
# p_random = 0
agent_loc = "saved_models/her_mod_"
# agent_loc = "saved_models/her_"


def set_state(self, state):
    self.time_limit = 100
    self.pos = np.array(state)[...,:2]
    self.vel = np.array(state)[...,2:]


class Car: 
    def __init__(self):
        self.env = CarEnvironment("Car")
        self.setup()

        if use_agent: 
            agent = DDPGAgentWrapper(agent_loc + "Car.pkl", goal_conditioned=True)
            def make_control_selector(controlSpace,metric,numSamples):
                return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = p_goal, p_random=1., goal=self.goal)
                # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = 0, p_random=1, goal=goal)
            self.control_space.controlSelector = make_control_selector

    def setup(self):
        obs = self.env.reset()
        self.start_state = obs['observation']#.tolist()
        self.goal = obs['desired_goal'].tolist()
        setattr(self.env, 'set_state', set_state)
        self.control_space = GymWrapperGoalConditionedControlSpace(self.env, self.goal)

    def controlSpace(self):
        return self.control_space

    def controlSet(self):
        return self.control_space.action_set

    def startState(self):
        return self.start_state.tolist()

    def configurationSpace(self):
        return self.control_space.configuration_space
        # res =  MultiConfigurationSpace(SO2Space(),BoxConfigurationSpace([self.omega_min],[self.omega_max]))
        #res.setDistanceWeights([1.0/(2*math.pi),1.0/(self.omega_max-self.omega_min)])
        # return res
    
    def goalSet(self):
        return self.control_space.goal_set



# def fetch_reach_test():
#     print("Testing gym pendulum")
#     p = Momentum()
#     configurationspace = p.configurationSpace()
#     assert len(configurationspace.sample()) == 4
#     assert configurationspace.contains([0,0,0,0])

#     controlset = p.controlSet()
#     assert len(controlset.sample()) == 3
#     assert controlset.contains([1,0,0])


def carTest():
    # gym_momentum_test()
    p = Car()
    objective = TimeObjectiveFunction()
    controlSpace = p.controlSpace()
    startState = p.startState()
    goalSet = p.goalSet()
    return PlanningProblem(controlSpace, startState, goalSet,
                           objective=objective)
