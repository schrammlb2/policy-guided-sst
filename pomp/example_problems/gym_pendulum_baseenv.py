import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path



class PendulumGoalEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)

        # if (reward > -.5).any(): 
        #     import pdb
        #     pdb.set_trace()

        self.state = np.array([newth, newthdot])
        return obs, reward, False, {'is_success': reward > -.5}
        # return obs, reward/((2*np.pi)**2 + 2**2), False, {'is_success': reward > -.5}
        # return obs['observation'], reward/10, False, {'is_success': reward > -.1}
        # return obs['observation'], reward, False, {'is_success': reward > -.1}
        # return obs['observation'], reward  + .001 * (u ** 2), False, {'is_success': reward > -1}
        # return obs['observation'], reward  + .0001 * (u ** 2), False, {'is_success': reward > -1}
        # return obs['observation'], -costs, False, {'is_success': reward > -1}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()#['observation']

    def _get_obs(self):
        state = self._get_state()
        theta, thetadot = self.state
        return {'observation': np.array([np.cos(theta), np.sin(theta), thetadot]), 'achieved_goal': self.state, 'desired_goal': np.zeros(2)}
        # return {'observation': state, 'achieved_goal': self.state, 'desired_goal': np.zeros(2)}
        # return {'observation': state, 'achieved_goal': np.array([self.state[0]]), 'desired_goal': np.zeros(1)}
        # return {'observation': state, 'achieved_goal': state[:2], 'desired_goal': np.array([0.,1.])}
        # return {'observation': state, 'achieved_goal': state, 'desired_goal': np.array([0.,1.,0])}

    def _get_state(self):        
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def compute_reward(self, a, b, info):
        # b = a*0
        # dist = ((a[...,:2] - b[...,:2])**2).sum(axis=-1)
        # dist = ((a - b)**2).sum(axis=-1)
        # dist = ((a[...,:2] - b[...,:2])**2).sum(axis=-1) + .1*((a[...,2:] - b[...,2:])**2).sum(axis=-1)
        # dist = ((a[...,1] - b[...,1])**2) + .1*((a[...,2:] - b[...,2:])**2).sum(axis=-1)
        # dist = (a[...,0] - b[...,0])**2 + .1*(a[...,1] - b[...,1])**2# + .1*((a[...,2:] - b[...,2:])**2).sum(axis=-1)
        dist = (angle_normalize(a[...,0]) - angle_normalize(b[...,0]))**2 + .1*(a[...,1] - b[...,1])**2

        return -dist#/((2*np.pi)**2 + 2**2)#/10
        # return  (dist < .5) - 1

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




# class PendulumGoalEnv(gym.GoalEnv):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 30
#     }

#     def __init__(self, g=10.0):
#         self.max_speed = 8
#         self.max_torque = 2.
#         self.dt = .05
#         self.g = g
#         self.m = 1.
#         self.l = 1.
#         self.viewer = None

#         high = np.array([1., 1., self.max_speed], dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=-self.max_torque,
#             high=self.max_torque, shape=(1,),
#             dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=-high,
#             high=high,
#             dtype=np.float32
#         )

#         self.seed()

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def step(self, u):
#         th, thdot = self.state  # th := theta

#         g = self.g
#         m = self.m
#         l = self.l
#         dt = self.dt

#         u = np.clip(u, -self.max_torque, self.max_torque)[0]
#         self.last_u = u  # for rendering
#         costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

#         newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
#         newth = th + newthdot * dt
#         newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

#         self.state = np.array([newth, newthdot])
#         return self._get_obs(), -costs, False, {}

#     def reset(self):
#         high = np.array([np.pi, 1])
#         self.state = self.np_random.uniform(low=-high, high=high)
#         self.last_u = None
#         return self._get_obs()

#     def _get_obs(self):
#         theta, thetadot = self.state
#         return np.array([np.cos(theta), np.sin(theta), thetadot])

#     def render(self, mode='human'):
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(500, 500)
#             self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
#             rod = rendering.make_capsule(1, .2)
#             rod.set_color(.8, .3, .3)
#             self.pole_transform = rendering.Transform()
#             rod.add_attr(self.pole_transform)
#             self.viewer.add_geom(rod)
#             axle = rendering.make_circle(.05)
#             axle.set_color(0, 0, 0)
#             self.viewer.add_geom(axle)
#             fname = path.join(path.dirname(__file__), "assets/clockwise.png")
#             self.img = rendering.Image(fname, 1., 1.)
#             self.imgtrans = rendering.Transform()
#             self.img.add_attr(self.imgtrans)

#         self.viewer.add_onetime(self.img)
#         self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
#         if self.last_u:
#             self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

#         return self.viewer.render(return_rgb_array=mode == 'rgb_array')

#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None





class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None









class PendulumEnvSO2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        # high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        # self.observation_space = spaces.Box(
        #     low=-high,
        #     high=high,
        #     dtype=np.float32
        # )

        high = np.array([np.pi,  self.max_speed], dtype=np.float32)
        low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
        return np.array([theta, thetadot])



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
