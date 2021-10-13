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
from HER_mod.rl_modules.car_env import RotationEnv, square_to_radian, radian_to_square
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



# def pendulum_set_state(self, state):
#     theta = math.asin(state[1])
#     theta_dot = state[-1]
#     # self.env.state = [theta, theta_dot]
#     self.state = [theta, theta_dot]

def set_state(self, state):
    self.position = np.array(state[:2])
    self.velocity = np.array(state[2:4])
    self.rotation = square_to_radian(np.array(state[-1:]))

# agent_loc = "saved_models/her_mod_"
agent_loc = "saved_models/her_"
goal_suffix = ".pkl"
p2p_suffix = "_p2p.pkl"

heuristic_infix = "_distance"
value_function_infix = "_value"

mean_GD_steps = 5
epsilon = 1/(1+mean_GD_steps)

# env_name = "Asteroids"
p2p_name = "AsteroidsVelGoal"

class Asteroids: 
    def __init__(self, shift=False, vel_goal = False, shift_scale=1):
        # self.env = MultiGoalEnvironment("MultiGoalEnvironment", vel_goal=True, time=True)
        self.env = RotationEnv(shift=shift, vel_goal=False, shift_scale=1)
        observation = self.env.reset()
        setattr(self.env, 'set_state', set_state)
        # goal = [0,0,-.9,-.9]
        goal = observation['desired_goal'].tolist()
        self.start_state = observation['observation'].tolist()
        # self.start_state = [-.95, -.95, 0., 0.0]
        self.goal = goal
        self.control_space = GymWrapperGoalConditionedControlSpace(self.env, goal)


        env_name = "Asteroids" + ("VelGoal" if vel_goal else "")
        pure_rl_agent = DDPGAgentWrapper(agent_loc + env_name + goal_suffix, goal_conditioned=True, deterministic=True)
        agent = DDPGAgentWrapper(agent_loc + env_name + goal_suffix, goal_conditioned=True)
        # p2p_agent = DDPGAgentWrapper(agent_loc + env_name + p2p_suffix, goal_conditioned=True)
        p2p_agent = DDPGAgentWrapper(agent_loc + p2p_name + goal_suffix, goal_conditioned=True)
        # def make_control_selector(controlSpace,metric,numSamples):
        #     return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = .5, p_random=.2, goal=goal)
        
        def pure_rl_selector(controlSpace,metric,numSamples):
            return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = pure_rl_agent,
                p_goal = 1, p_random=0, goal=self.goal)

        def control_selector_maker(p_goal, p_random):
            rv = lambda controlSpace,metric,numSamples: RLAgentControlSelector(controlSpace,metric,numSamples, 
                rl_agent = agent, p2p_agent=p2p_agent, p_goal = p_goal, p_random=p_random, goal=goal)
            return rv

        # self.control_space.controlSelector = make_control_selector
        # self.control_space.controlSelector = control_selector_maker(.5, .2)
        self.control_space.controlSelector = control_selector_maker(.2, .5)
        # self.control_space.controlSelector = control_selector_maker(.5, .5)
        self.control_space.p2pControlSelector = control_selector_maker(0, 0)
        self.control_space.pure_rl_controlSelector = pure_rl_selector
        self.set_heuristic(env_name)

        self.set_value_function(env_name)

    def controlSpace(self):
        return self.control_space

    def controlSet(self):
        return self.control_space.action_set

    def startState(self):
        return self.start_state#[-.95, -.95, 0., 0.0]

    def configurationSpace(self):
        return self.control_space.configuration_space
        # res =  MultiConfigurationSpace(SO2Space(),BoxConfigurationSpace([self.omega_min],[self.omega_max]))
        #res.setDistanceWeights([1.0/(2*math.pi),1.0/(self.omega_max-self.omega_min)])
        # return res
    
    def goalSet(self):
        return self.control_space.goal_set

    def set_heuristic(self, heuristic_name):
        heuristic = HeuristicWrapper(agent_loc + heuristic_name + heuristic_infix + p2p_suffix, self.goal, goal_conditioned=True)
        # def make_control_selector(controlSpace,metric,numSamples):
        #     return lambda x: heuristic.evaluate(x, self.goal)
            # return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = 0, p_random=1, goal=goal)
        heuristic.evaluate = lambda a, b: 0
        self.control_space.heuristic = heuristic

    def set_value_function(self, value_function_name):
        cs = self.configurationSpace()
        goal_filename = agent_loc + value_function_name + value_function_infix + goal_suffix
        with open(goal_filename, 'rb') as f:
            goal_value = pickle.load(f)
        # p2p_filename = agent_loc + value_function_name + value_function_infix + p2p_suffix
        p2p_filename = agent_loc + p2p_name + value_function_infix + goal_suffix
        with open(p2p_filename, 'rb') as f:
            p2p_value = pickle.load(f)

        start_state = self.start_state
        goal = self.goal

        # value = lambda x: goal_value(x) + p2p_value(x)**.5
        # gd_sampler = GDValueSampler(cs, goal_value, start_state, goal, epsilon=epsilon)
        gd_sampler = GDValueSampler(cs, goal_value, p2p_value, start_state, goal, epsilon=epsilon, 
            norm=goal_value.actor._get_norms, denorm=goal_value.actor._get_denorms)


        def normed_sample():
            samp = torch.tensor(np.random.randn(start_state.shape[0]))#*2
            return goal_value.actor._get_denorms(samp, torch.tensor(goal))[0].tolist()

        # setattr(self.control_space.configuration_space, 'distance', dist)
        # setattr(self.control_space.configuration_space, 'sample', normed_sample)
        # gd_sampler = GDValueSampler(cs, value, start_state, goal, epsilon=epsilon)
        # self.control_space.configuration_space = gd_sampler
        self.control_space.configuration_sampler = gd_sampler


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


def gymAsteroidsTest():
    # gym_momentum_test()
    p = Asteroids()
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)


def gymAsteroidsVelGoalTest():
    # gym_momentum_test()
    p = Asteroids(vel_goal=True)
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)


def gymAsteroidsShiftTest():
    # gym_momentum_test()
    p = Asteroids(shift=True)
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)

def gymAsteroidsShiftTest2():
    # gym_momentum_test()
    p = Asteroids(shift=True, shift_scale=2)
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)


def gymAsteroidsShiftTest3():
    # gym_momentum_test()
    p = Asteroids(shift=True, shift_scale=3)
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)


def gymAsteroidsShiftTest4():
    # gym_momentum_test()
    p = Asteroids(shift=True, shift_scale=4)
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)