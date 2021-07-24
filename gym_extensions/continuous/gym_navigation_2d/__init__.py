# from .range_based_navigation import LimitedRangeBasedPOMDPNavigation2DEnv, StateBasedMDPNavigation2DEnv
from .goal_conditioned_env import LimitedRangeBasedPOMDPNavigation2DEnv, StateBasedMDPNavigation2DEnv
from .image_based_navigation import ImageBasedNavigation2DEnv
#
from gym.envs import register
import numpy as np

# (x,y)
n_goals = 10
# idx_to_goal = [ (np.array([230.0, 430.0]), np.array([230.0, 130.0]), np.array([500.0, 430.0]) )] * n_goals
# idx_to_goal[1] = (np.array([130.0, 370.0]), np.array([130.0, 110.0]), np.array([520.0, 250.0]) )
# idx_to_goal[2] = (np.array([530.0, 110.0]), np.array([130.0, 310.0]), np.array([460.0, 330.0]) )
# idx_to_goal[3] = (np.array([400.0, 50.0]), np.array([180.0, 320.0]), np.array([430.0, 310.0]) )
# idx_to_goal[4] = (np.array([180.0, 380.0]), np.array([610.0, 120.0]), np.array([420.0, 330.0]) )
# idx_to_goal[5] = (np.array([500.0, 90.0]), np.array([180.0, 390.0]), np.array([380.0, 320.0]) )
# idx_to_goal[6] = (np.array([480.0, 150.0]), np.array([440.0, 380.0]), np.array([310.0, 220.0]) )
# idx_to_goal[7] = (np.array([500.0, 380.0]), np.array([470.0, 280.0]), np.array([270.0, 280.0]) )
# idx_to_goal[8] = (np.array([250.0, 440.0]), np.array([420.0, 200.0]), np.array([150.0, 180.0]) )
# idx_to_goal[9] = (np.array([390.0, 110.0]), np.array([520.0, 350.0]), np.array([240.0, 310.0]) )
idx_to_goal = [[np.random.randn(2)*0-20 for i in range(3)] for j in range(n_goals)]

custom_envs = {}

max_episode_steps=200

for i in range(n_goals):
    for j in range(3):
        # add each env to dictionary       
        for info, problem_type in [('State', 'StateBasedMDP'), ('Limited-Range', 'LimitedRangeBasedPOMDP')]: 
            custom_envs[info + '-Based-Navigation-2d-Map%d-Goal%d-v0' % (i, j)] = dict(
                path='gym_extensions.continuous.gym_navigation_2d:' + problem_type + 'Navigation2DEnv',
                max_episode_steps=max_episode_steps,
                kwargs=dict(world_idx=i, destination = idx_to_goal[i][j], 
                    add_self_position_to_observation=True))

            custom_envs[info + '-Based-Navigation-2d-Map%d-Goal%d-KnownPositions-v0' % (i, j)] = dict(
                path='gym_extensions.continuous.gym_navigation_2d:' + problem_type + 'Navigation2DEnv',
                max_episode_steps=max_episode_steps,
                kwargs=dict(world_idx=i, destination = idx_to_goal[i][j], 
                    add_self_position_to_observation=True, 
                    add_goal_position_to_observation=True))

        # custom_envs['State-Based-Navigation-2d-Map%d-Goal%d-v0' % (i, j)] = dict(
        #          path='gym_extensions.continuous.gym_navigation_2d:StateBasedMDPNavigation2DEnv',
        #          max_episode_steps=1000, 
        #          kwargs=dict(world_idx=i, destination=idx_to_goal[i][j]))

        # # custom_envs['State-Based-Navigation-2d-Map%d-Goal%d-KnownGoalPosition-v0' % (i, j)] = dict(
        # #          path='gym_extensions.continuous.gym_navigation_2d:StateBasedMDPNavigation2DEnv',
        # #          max_episode_steps=1000,
        # #          kwargs=dict(world_idx=i, destination = idx_to_goal[i][j]))

        # custom_envs['State-Based-Navigation-2d-Map%d-Goal%d-KnownGoalPosition-v0' % (i, j)] = dict(
        #     path='gym_extensions.continuous.gym_navigation_2d:StateBasedMDPNavigation2DEnv',
        #     max_episode_steps=1000,
        #     kwargs=dict(world_idx=i, destination = idx_to_goal[i][j], 
        #                 add_self_position_to_observation=True, 
        #                 add_goal_position_to_observation=True))

        # custom_envs['Limited-Range-Based-Navigation-2d-Map%d-Goal%d-v0' % (i, j)] = dict(
        #     path='gym_extensions.continuous.gym_navigation_2d:LimitedRangeBasedPOMDPNavigation2DEnv',
        #     max_episode_steps=1000,
        #     kwargs=dict(world_idx=i, destination = idx_to_goal[i][j]))

        # custom_envs['Limited-Range-Based-Navigation-2d-Map%d-Goal%d-KnownPositions-v0' % (i, j)] = dict(
        #     path='gym_extensions.continuous.gym_navigation_2d:LimitedRangeBasedPOMDPNavigation2DEnv',
        #     max_episode_steps=1000,
        #     kwargs=dict(world_idx=i, destination = idx_to_goal[i][j], 
        #         add_self_position_to_observation=True, 
        #         add_goal_position_to_observation=True))

        custom_envs['Image-Based-Navigation-2d-Map%d-Goal%d-v0' % (i, j)] = dict(
            path='gym_extensions.continuous.gym_navigation_2d:ImageBasedNavigation2DEnv',
            # max_episode_steps=1000,
            max_episode_steps=max_episode_steps,
            kwargs=dict(world_idx=i, destination = idx_to_goal[i][j]))

# register each env into 
def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key, 
                        entry_point=value['path'], 
                        max_episode_steps=value['max_episode_steps'], 
                        kwargs=value['kwargs'])
        if 'reward_threshold' in value.keys():
            arg_dict['reward_threshold'] = value['reward_threshold']
        register(**arg_dict)

register_custom_envs()
