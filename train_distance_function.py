
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

from collect_distance_data import my_dataset

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
	parser.add_argument('--dropout-rate', type=float, default=.2, help='dropout rate for dropout layers')
	parser.add_argument('--p2p', action='store_true', help='if we\'re using the p2p agent')


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



class net(nn.Module):
	def __init__(self, input_shape, args):
		super(net, self).__init__()
		# self.norm1 = nn.LayerNorm(256)
		# self.norm2 = nn.LayerNorm(256)
		# self.norm3 = nn.LayerNorm(256)
		self.norm1 = nn.BatchNorm1d(input_shape)
		# self.norm1 = nn.BatchNorm1d(args.hidden_size)
		self.norm2 = nn.BatchNorm1d(args.hidden_size)
		self.norm3 = nn.BatchNorm1d(args.hidden_size)
		self.norm4 = nn.BatchNorm1d(args.hidden_size)
		self.fc1 = nn.Linear(input_shape, args.hidden_size)
		self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
		self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
		self.q_out = nn.Linear(args.hidden_size, 1)
		self.do1 = nn.Dropout(args.dropout_rate)
		self.do2 = nn.Dropout(args.dropout_rate)
		self.do3 = nn.Dropout(args.dropout_rate)
		# self.do1 = nn.AlphaDropout(args.dropout_rate)
		# self.do2 = nn.AlphaDropout(args.dropout_rate)
		# self.do3 = nn.AlphaDropout(args.dropout_rate)
		self.act = F.relu
		# self.act = F.selu


	# def forward(self, x):
	#     x = self.norm1(x)
	#     x = F.relu(self.fc1(x))
	#     x = self.norm2(x)
	#     x = self.do1(x)
	#     x = F.relu(self.fc2(x))
	#     x = self.norm3(x)
	#     x = self.do2(x)
	#     x = F.relu(self.fc3(x))
	#     x = self.do3(x)
	#     q_value = self.q_out(x)

	#     return q_value

	def forward(self, x):
		self.act = F.relu
		x = self.norm1(x)
		x = self.act(self.fc1(x))
		x = self.norm2(x)
		x = self.do1(x)
		x = self.act(self.fc2(x))
		x = self.norm3(x)
		x = self.do2(x)
		x = self.act(self.fc3(x))
		x = self.norm4(x)
		x = self.do3(x)
		q_value = self.q_out(x)

		return q_value

def train(args, model, training_dataloader, testing_dataloader): 
	opt = torch.optim.Adam(model.parameters(), lr=.001)
	for i in range(args.epochs):
		if i % max(args.epochs//10, 1) == 0: 
			print("Epoch: " + str(i))
		train_l = []
		model.train()
		for train_features, train_labels in training_dataloader:
			if args.cuda: 
				train_features=train_features.cuda()
				train_labels=train_labels.cuda()
			train_out = model(train_features)
			l = ((train_labels - train_out)**2).mean()
			opt.zero_grad()
			l.backward()
			train_l.append(l.detach())
			opt.step()
		train_loss = sum(train_l)/len(train_l)

		if i % max(args.epochs//10, 1) == 0: 
			print("Training loss: " + str(train_loss))


		model.eval()
		test_l = []
		for test_features, test_labels in testing_dataloader:
			if args.cuda: 
				test_features=test_features.cuda()
				test_labels=test_labels.cuda()
			out = model(test_features)
			l = ((test_labels - out)**2).mean()
			test_l.append(l.detach())
		test_loss = sum(test_l)/len(test_l)
		if i % max(args.epochs//10, 1) == 0: 
			print("Testing loss: " + str(test_loss))


# args = get_args()
# agent = get_agent(args.agent_location)

# data, labels, input_shape = collect_data_goal_env(agent, args)
# training_dataset = my_dataset(data[:-2*len(data)//10], labels[:-2*len(labels)//10])
# testing_dataset  = my_dataset(data[-2*len(data)//10:], labels[-2*len(labels)//10:])

if __name__ == "__main__": 
	args = get_args()
	if args.p2p: 
		suffix = '_p2p.pkl'
	else: 
		suffix = '.pkl'

	with open('datasets/train_' + args.env_name + suffix, 'rb') as f:
		training_dataset = pickle.load(f)

	with open('datasets/test_' + args.env_name + suffix, 'rb') as f:
		testing_dataset = pickle.load(f)

	train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
	test_dataloader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

	env = make_env(args)
	obs = env.reset()

	if args.p2p: 
		input_shape = 2*obs['observation'].shape[0] 
	else: 
		input_shape = obs['observation'].shape[0] + obs['desired_goal'].shape[0]



	model = net(input_shape, args)

	if args.cuda: 
		model.cuda()

	train(args, model, train_dataloader, test_dataloader)
	model.cpu()
	with open(args.agent_location[:-4] + '_distance' + suffix, 'wb') as f:
	    pickle.dump(model, f)
	    print("Saved models")