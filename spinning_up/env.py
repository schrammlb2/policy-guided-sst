import gym
import torch
# import pybullet
# import pybulletgym

import numpy as np
from gym import utils
from spinning_up.hyperparams import *
# from modified_envs import *
from gym import wrappers
from time import time

# class Env():
#   def __init__(self, name = 'Pendulum-v0', xml_file=None, record=False, algorithm_name=''):
#     self.name = name
#     if xml_file is None:
#       self._env = gym.make(name)
#     else: 
#       try:
#         self._env = gym.make(name, xml_file=xml_file)
#       except:
#         if 'InvertedPendulum' in name:
#           self._env=InvertedPendulumEnv_Mod(xml_file)
#         else:
#           #Other environments go here
#           pass
#     if record:
#       self._env = wrappers.Monitor(self._env, './videos/' + algorithm_name + '_' + name + '_' + str(time()) + '/')

#   def reset(self):
#     state = self._env.reset()
#     return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0)
  
#   def step(self, action):
#     state, reward, done, _ = self._env.step(action[0].detach().cpu().numpy())
#     return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0), reward, done

#   def sample_action(self):
#     return torch.tensor(self._env.action_space.sample(), device=DEVICE).unsqueeze(dim=0)

#   # def modify(self, xml_file):
#   #   self._env = gym.make(self.name, xml_file=xml_file)
#   def render(self):
#     self._env.render()

#   def close(self):
#     self._env.close()

class Env():
  def __init__(self, env, xml_file=None, record=False, algorithm_name=''):
    self._env = env

  def reset(self):
    state = self._env.reset()
    return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0)
  
  def step(self, action):
    state, reward, done, _ = self._env.step(action[0].detach().cpu().numpy())
    return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0), reward, done

  def sample_action(self):
    return torch.tensor(self._env.action_space.sample(), device=DEVICE).unsqueeze(dim=0)

  # def modify(self, xml_file):
  #   self._env = gym.make(self.name, xml_file=xml_file)
  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

def record_run(env_name, agent, algorithm_name=''):
  for _ in range(4):
    env = Env(name=env_name, record=True, algorithm_name=algorithm_name)
    state = env.reset()
    done=False
    while not done:
      env.render()
      policy, value = agent(state)
      action = policy.sample()
      next_state, reward, done = env.step(action)
      state = next_state
  env.close()


