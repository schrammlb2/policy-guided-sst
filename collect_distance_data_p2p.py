
import argparse
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
import numpy as np

import gym
from gym.wrappers.time_limit import TimeLimit

import pickle

from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv
from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
from pomp.example_problems.robotics.fetch.push import FetchPushEnv
from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from HER_mod.rl_modules.velocity_env import MultiGoalEnvironment, CarEnvironment
from HER_mod.rl_modules.car_env import *


from sample_valid_goal import sample_valid_goal
from pomp.example_problems.robotics.hand.reach import HandReachEnv

# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment, EnvironmentCollection, Obstacle
from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle


num_episodes = 5000
epochs = 500

num_episodes = None
epochs = None

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
	parser.add_argument('--max-steps', type=int, default=None, help='number of steps allowed in rollout')
	parser.add_argument('--agent-location', type=str, default="saved_models/her_FetchReach.pkl", help='location the agent is stored')
	parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
	parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
	parser.add_argument('--hidden-size', type=int, default=256, help='the sample batch size')
	parser.add_argument('--episodes', type=int, default=5000, help='number of episodes to sample for data')
	parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')


	args = parser.parse_args()
	return args

def get_agent(filename):
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    return agent



def make_env(args):
    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif "Car" in args.env_name:
        env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif args.env_name == "Asteroids" :
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "AsteroidsVelGoal" :
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    elif "FetchReach" in args.env_name:
        env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    elif "FetchPush" in args.env_name:
        env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    elif "FetchSlide" in args.env_name:
        env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
    elif "FetchPickAndPlace" in args.env_name:
        env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
    elif args.env_name == "HandReach":
        env = TimeLimit(HandReachEnv(), max_episode_steps=10)
    else:
        env = gym.make(args.env_name)

    return env




def collect_data_goal_env(agent, args):
	env = make_env(args)
	if args.max_steps == None: 
		max_steps = env._max_episode_steps
	else: 
		max_steps = args.max_steps
	done = False
	data = []
	labels = []
	successes = 0
	failures = 0
	reward_fn = lambda o, g, x=None: -np.mean((o-g)**2, axis=-1)
	for i in range(1, args.episodes + 1):
		if i % (args.episodes//10) == 0: 
			print("Episode:" + str(i))
			print("Success rate: " + str(successes/(successes+ failures)))
		episode = []
		obs = env.reset()
		g = env.observation_space['observation'].sample()
		for _ in range(max_steps):
			o = obs['observation']
			action = agent.normed_forward(o,g, deterministic=True).detach().numpy().squeeze(axis=0)
			obs_new, reward, _, info = env.step(action) # take a random action
			reward = reward_fn(obs_new['observation'], g)
			episode.append((obs_new, reward))
			obs = obs_new

		cum_r = 0
		for time in reversed(episode):
			if info['is_success'] == True:
				cum_r -= time[1] 
				successes += 1
			else: 
				cum_r = max_steps
				failures += 1
			# cum_r = time[1] 
			obs = time[0]
			# cum_r = env.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
			x = torch.tensor(np.concatenate([obs['observation'], g]),dtype=torch.float32)
			# x = torch.tensor(np.concatenate([obs['achieved_goal'], obs['desired_goal']]),dtype=torch.float32)
			# diff_sum = (torch.tensor([(obs['achieved_goal']- obs['desired_goal'])**2],dtype=torch.float32).sum(dim=-1))#*2./.0025 - 1
			# x.cuda()
			data.append(x)
			# data.append(diff_sum)

			r_tensor = torch.tensor([cum_r],dtype=torch.float32)
			labels.append(r_tensor)

	env.close()
	# return data, labels, (obs['achieved_goal'].shape[0] + obs['desired_goal'].shape[0])
	# return data, labels, (1)
	return data, labels, (obs['observation'].shape[0] + obs['desired_goal'].shape[0])

class my_dataset(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label          
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)



if __name__ == "__main__": 
	args = get_args()
	agent = get_agent(args.agent_location)

	data, labels, input_shape = collect_data_goal_env(agent, args)	
	training_dataset = my_dataset(data[:-2*len(data)//10], labels[:-2*len(labels)//10])
	testing_dataset  = my_dataset(data[-2*len(data)//10:], labels[-2*len(labels)//10:])

	with open('datasets/train_' + args.env_name + '_p2p.pkl', 'wb') as f:
		pickle.dump(training_dataset, f)
		print("Saved models")

	with open('datasets/test_' + args.env_name + '_p2p.pkl', 'wb') as f:
		pickle.dump(testing_dataset, f)
		print("Saved models")

# train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
# test_dataloader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

# model = net(input_shape, args)
# # model = nn.Sequential(
# #           nn.Linear(input_shape,256),
# #           nn.SELU(),
# #           nn.Linear(256,256),
# #           nn.SELU(),
# #           nn.Linear(256,1)
# #         )
# if args.cuda: 
# 	# train_dataloader.cuda()
# 	# test_dataloader.cuda()
# 	model.cuda()
# train(args, model, train_dataloader, test_dataloader)
# with open('datasets/train_' + args.env_name + '.pkl', 'wb') as f:
#     pickle.dump(model, f)
#     print("Saved models")