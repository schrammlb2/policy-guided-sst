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
from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment
import math
import pickle
import gym


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


# agent_loc = "saved_models/her_mod_"
agent_loc = "saved_models/her_"
goal_suffix = ".pkl"
p2p_suffix = "_p2p.pkl"
# p2p_name = "MultiGoalEnvironmentVelGoal"

heuristic_infix = "_distance"
value_function_infix = "_value"

mean_GD_steps = 5
epsilon = 1/(1+mean_GD_steps)
# class GymPendulum: 
use_agent = True

env = 'lidar'

class Obstacle2D: 
    def __init__(self, shift=False, vel_goal=False, env='lidar'):
        
        if env == 'lidar':
            env_name = "Limited-Range-Based-Navigation-2d-Map4-Goal0-v0"
            def set_state(self, state):
                self.state = np.array(state[:2])
                # self.observation = self._get_observation(self.state)

        else:
            env_name = "State-Based-Navigation-2d-Map4-Goal0-v0"
            def set_state(self, state):
                self.state = np.array(state[:2])
                # self.observation = self._get_observation(self.state)
        self.env = gym.make(env_name).env
        observation = self.env.reset()
        setattr(self.env, 'set_state', set_state)
        # self.start_state = [-.95, -.95, 0., 0.0]
        # goal = [0,0]#,-.9,-.9]
        # self.env.goal = np.array(goal)
        # self.env.set_state(self.env, self.start_state)
        goal = observation['desired_goal'].tolist()
        self.start_state = observation['observation'].tolist()
        self.goal = goal

        self.control_space = GymWrapperGoalConditionedControlSpace(self.env, goal)

        env_name = env_name + ("VelGoal" if vel_goal else "")
        if use_agent: 
            pure_rl_agent = DDPGAgentWrapper(agent_loc + env_name + goal_suffix, goal_conditioned=True, deterministic=True)
            agent = DDPGAgentWrapper(agent_loc + env_name + goal_suffix, goal_conditioned=True)
            p2p_agent = DDPGAgentWrapper(agent_loc + env_name + goal_suffix, goal_conditioned=True)
            # p2p_agent = DDPGAgentWrapper(agent_loc + env_name + p2p_suffix, goal_conditioned=True)
            # p2p_agent = DDPGAgentWrapper(agent_loc + p2p_name + goal_suffix, goal_conditioned=True)
            # p2p_agent = None
            # def make_control_selector(controlSpace,metric,numSamples):
            #     return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = agent, p_goal = .5, p_random=.2, goal=goal)
            def pure_rl_selector(controlSpace,metric,numSamples):
                return RLAgentControlSelector(controlSpace,metric,numSamples, rl_agent = pure_rl_agent,
                    p_goal = 1, p_random=0, goal=self.goal)

            def control_selector_maker(p_goal, p_random):
                rv = lambda controlSpace,metric,numSamples: RLAgentControlSelector(controlSpace,metric,numSamples, 
                    rl_agent = agent, p2p_agent=agent, p_goal = p_goal, p_random=p_random, goal=goal)
                return rv

            # self.control_space.controlSelector = make_control_selector
            self.control_space.controlSelector = control_selector_maker(.5,.5)
            # self.control_space.controlSelector = control_selector_maker(.3,.3)
            # self.control_space.controlSelector = control_selector_maker(.2, .5)
            # self.control_space.controlSelector = control_selector_maker(.5, .5)
            # self.control_space.controlSelector = control_selector_maker(.0, .5)
            self.control_space.p2pControlSelector = control_selector_maker(0, 0)
            self.control_space.pure_rl_controlSelector = pure_rl_selector

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

    def set_value_function(self, value_function_name):
        cs = self.configurationSpace()
        goal_filename = agent_loc + value_function_name + value_function_infix + goal_suffix
        with open(goal_filename, 'rb') as f:
            goal_value = pickle.load(f)
        # p2p_filename = agent_loc + value_function_name + value_function_infix + p2p_suffix
        p2p_filename = goal_filename
        with open(p2p_filename, 'rb') as f:
            p2p_value = pickle.load(f)
        # p2p_value = None

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


def gym_obstacle_test():
    print("Testing gym obstacle problem")
    p = Obstacle2D()
    configurationspace = p.configurationSpace()
    controlset = p.controlSet()
    env = p.controlSpace().env
    env.reset()
    ag_list = []
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        ag_list.append(obs['achieved_goal'])
    print(ag_list)
    env._render(mode='human', path=ag_list)


def gymObstacle2DLidarTest():
    # gym_obstacle_test()
    p = Obstacle2D()
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)

def gymObstacle2DTest():
    # gym_obstacle_test()
    p = Obstacle2D(env='state')
    # objective = TimeObjectiveFunction()
    objective = TimeLengthObjectiveFunction()
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)