import copy 
import gym
import numpy as np

def sample_valid_goal(base_env):
    if np.random.binomial(1, .5):
        return base_env.observation_space['observation'].sample()

    env = copy.deepcopy(base_env)
    env.reset()
    step_num = int(.9*env._max_episode_steps)
    step_num = 25
    for i in range(step_num):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action*0)

    return observation['observation']
