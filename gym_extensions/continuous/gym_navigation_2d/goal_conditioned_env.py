import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Tuple
from .env_generator import EnvironmentCollection

from math import pi, cos, sin
import numpy as np

#from gym.envs.classic_control.rendering  import make_circle, Transform
from gym_extensions.continuous.gym_navigation_2d import gym_rendering  
import os
import logging 

class LimitedRangeBasedPOMDPNavigation2DEnv(gym.GoalEnv):
    logger = logging.getLogger(__name__)
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v0.pkl"),
                 world_idx=0,
                 initial_position = np.array([-20.0, -20.0]),
                 destination = np.array([520.0, 400.0]),
                 max_observation_range = 100.0,
                 destination_tolerance_range=20.0,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False):

        worlds = EnvironmentCollection()
        worlds.read(worlds_pickle_filename)

        self.world = worlds.map_collection[world_idx]
        self.set_destination(destination)

        assert not (self.destination is None)
        self.init_position = initial_position
        self.state = self.init_position.copy()


        self.max_observation_range = max_observation_range
        self.destination_tolerance_range = destination_tolerance_range
        self.viewer = None
        self.num_beams = 16
        self.max_speed = 5
        self.add_self_position_to_observation = add_self_position_to_observation
        self.add_goal_position_to_observation = add_goal_position_to_observation


        low = np.array([0.0, 0.0])
        high = np.array([self.max_speed, 2*pi])
        # self.action_space = Box(low, high)#Tuple( (Box(0.0, self.max_speed, (1,)), Box(0.0, 2*pi, (1,))) )
        self.action_space = Box(np.array([-1,-1]), np.array([1,1]))
        low = [-1.0] * self.num_beams
        high = [self.max_observation_range] * self.num_beams
        if add_self_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])
        if add_goal_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])

        xy_low = [-100., -100.]
        xy_high = [600., 600.]
        self.mod_ranges = True
        if self.mod_ranges:
            low = np.array(xy_low + [0.0] * self.num_beams)
        else: 
            low = np.array(xy_low + [-1.0] * self.num_beams)
        # high = np.array(xy_high + [self.max_observation_range] * self.num_beams)
        high = np.array(xy_high + [np.log(self.max_observation_range+1)] * self.num_beams)

        self._seed = 0

        self.observation_space = {'observation': Box(np.array(low), np.array(high)), 
            'achieved_goal': Box(np.array(xy_low), np.array(xy_high)),
            'desired_goal': Box(np.array(xy_low), np.array(xy_high)),
            }
        self.observation = []

    def set_destination(self, destination):
        self.destination = destination

    def _get_observation(self, state):
        delta_angle = 2*pi/self.num_beams
        ranges = [self.world.raytrace(self.state,
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(self.num_beams)]

        if self.mod_ranges:
            ranges = [np.log(self.max_observation_range+1) if r < 0 else np.log(r+1) for r in ranges]
        ranges = np.array(ranges)

        if self.add_self_position_to_observation:
            ranges = np.concatenate([self.state, ranges])
        if self.add_goal_position_to_observation:
            ranges = np.concatenate([ranges, self.destination])


        obs = {'observation': ranges, 
            'achieved_goal': state.copy(),
            'desired_goal': self.destination.copy(), 
        }
        return obs
        
    def _get_obs(self):
        return self._get_observation(self.state)

    def _is_close(self, a, b):
        return np.linalg.norm(a - b, axis=-1) < self.destination_tolerance_range

    def compute_reward(self, state, goal, info):
        # return (1-self._is_close(state, goal))*(-np.log(np.linalg.norm(state - goal, axis=-1)/self.destination_tolerance_range+1))
        # return -np.linalg.norm(state - goal, axis=-1)/self.destination_tolerance_range
        return self._is_close(state, goal) - 1
        # if not self.world.point_is_in_free_space(state[0], state[1], epsilon=0.25):
        #     reward = -5
        # else: 
        #     reward = self._is_close(state, goal) - 1
        # return reward

    def _step(self, action):
        old_state = self.state.copy()
        # v = action[0]
        # theta = action[1]
        # dx = v*cos(theta)
        # dy = v*sin(theta)

        # self.state += np.array([dx, dy])
        new_state = self.state + action*self.max_speed

        # reward = -1 # minus 1 for every timestep you're not in the goal
        collided = False
        done = self._is_close(self.state, self.destination)
        info = {'is_success': done}

        # if np.linalg.norm(self.destination - self.state) < self.destination_tolerance_range:
        #     reward = 20 # for reaching the goal
        #     done = True

        if not self.world.point_is_in_free_space(new_state[0], new_state[1], epsilon=0.25):
            collision = True
            reward = -5 # for hitting an obstacle
        elif not self.world.segment_is_in_free_space(old_state[0], old_state[1],
                                                   new_state[0], new_state[1],
                                                   epsilon=0.25):
            collision = True
            reward = -5 # for hitting an obstacle
        else: 
            
            reward = self.compute_reward(self.state, self.destination, None)
            self.state = new_state
        
        obs = self._get_observation(self.state)
        obs['collided'] = collided
        self.observation = obs#['observation']
        return obs, reward, done, info


    def _reset(self):
        self.init_position = np.random.randn(2)*20
        self.state = self.init_position
        # sample_goal = lambda : np.random.randn(2)*20 + 50#np.random.randint(0, 500)
        sample_goal = lambda : np.array([np.random.randint(0, 500),np.random.randint(0, 400)])
        self.destination = sample_goal()
        while not self.world.point_is_in_free_space(self.destination[0], self.destination[1], epsilon=0.25):
            self.destination = sample_goal()

        self.goal = self.destination
        # self.destination = np.array([-20,-20])
        # print(self.destination)
        return self._get_observation(self.state)

    def _plot_state(self, viewer, state):
        polygon = gym_rendering.make_circle(radius=5, res=30, filled=True)
        state_tr = gym_rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(state_tr)
        viewer.add_onetime(polygon)

    def _plot_path(self, viewer, path):
        for state in path: 
            polygon = gym_rendering.make_circle(radius=2, res=30, filled=True)
            state_tr = gym_rendering.Transform(translation=(state[0], state[1]))
            polygon.add_attr(state_tr)
            polygon.set_color(0,0,255)
            viewer.add_onetime(polygon)


    def _plot_observation(self, viewer, state, observation):
        delta_angle = 2*pi/self.num_beams
        for i in range(len(observation)):
            r = observation[i]
            if r < 0:
                r = self.max_observation_range

            theta = i*delta_angle
            start = (state[0], state[1])
            end = (state[0] + r*cos(theta), state[1] + r*sin(theta))

            line = gym_rendering.Line(start=start, end=end)
            line.set_color(.5, 0.5, 0.5)
            viewer.add_onetime(line)

    def _append_elements_to_viewer(self, viewer,
                                   screen_width,
                                   screen_height,
                                   obstacles,
                                   destination=None,
                                   destination_tolerance_range=None):

        viewer.set_bounds(left=-100, right=screen_width+100, bottom=-100, top=screen_height+100)

        L = len(obstacles)
        for i in range(L):

            obs = obstacles[i]
            for c,w,h in zip(obs.rectangle_centers, obs.rectangle_widths, obs.rectangle_heights):
                l = -w/2.0
                r = w/2.0
                t = h/2.0
                b = -h/2.0

                rectangle = gym_rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                tr = gym_rendering.Transform(translation=(c[0], c[1]))
                rectangle.add_attr(tr)
                rectangle.set_color(.8,.6,.4)
                viewer.add_geom(rectangle)


        if not (destination is None):
            tr = gym_rendering.Transform(translation=(destination[0], destination[1]))
            polygon = gym_rendering.make_circle(radius=destination_tolerance_range, res=30, filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)

    def _render(self, mode='human', close=False, path=[]):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return

        screen_width = (self.world.x_range[1] - self.world.x_range[0])
        screen_height = (self.world.y_range[1] - self.world.y_range[0])

        if self.viewer is None:
            self.viewer = gym_rendering.Viewer(screen_width, screen_height)
            self._append_elements_to_viewer(self.viewer,
                                            screen_width,
                                            screen_height,
                                            obstacles=self.world.obstacles,
                                            destination=self.destination,
                                            destination_tolerance_range=self.destination_tolerance_range)

        self._plot_state(self.viewer, self.state)
        self._plot_path(self.viewer, path)
        # self._plot_observation(self.viewer, self.state, self.observation)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class StateBasedMDPNavigation2DEnv(LimitedRangeBasedPOMDPNavigation2DEnv):
    logger = logging.getLogger(__name__)
    def __init__(self, *args, **kwargs):
        LimitedRangeBasedPOMDPNavigation2DEnv.__init__(self, *args, **kwargs)
        low = [-float('inf'), -float('inf'), 0.0, 0.0]
        high = [float('inf'), float('inf'), float('inf'), 2*pi]

        xy_low = [-100., -100.]
        xy_high = [600., 600.]
        if self.add_goal_position_to_observation:
            low.extend(xy_low) # x and y coords
            high.extend(xy_high)

        # self.observation_space = Box(np.array(low), np.array(high))
        self.observation_space = {'observation': Box(np.array(xy_low + [-20,20]), np.array(xy_high + [20,20])), 
            'achieved_goal': Box(np.array(xy_low), np.array(xy_high)),
            'desired_goal': Box(np.array(xy_low), np.array(xy_high)),
            }
        self._seed = 0

    def _plot_observation(self, viewer, state, observation):
        pass

    def _get_observation(self, state):
        # return state
        reading = self.world.range_and_bearing_to_closest_obstacle(state[0], state[1])
        r = reading[0]/10
        theta = reading[1]
        dx = r*cos(theta)
        dy = r*sin(theta)
        # obs = np.array([state[0], state[1], dist_to_closest_obstacle, absolute_angle_to_closest_obstacle])
        obs = np.array([state[0], state[1], dx, dy])
        # if self.add_goal_position_to_observation:
        #     obs = np.concatenate([obs, self.destination])

        observation = {'observation': obs, 
            'achieved_goal': state.copy(),
            'desired_goal': self.destination.copy()
        }

        return observation
        # return obs
