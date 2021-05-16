import torch
from torch import optim
from tqdm import tqdm
from spinning_up.env import Env, record_run
from spinning_up.hyperparams import ON_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, PPO_CLIP_RATIO, PPO_EPOCHS, TRACE_DECAY
# from models import ActorCritic
# from utils import plot
from spinning_up.utils import plot, multiplot, get_rolling_ave


# env = Env()
# agent = ActorCritic(HIDDEN_SIZE)
from spinning_up.generalized_models import ActorCritic

from pomp.planners.plantogym import PlanningEnvGymWrapper, KinomaticGymWrapper, PendulumWrapper
from pomp.example_problems.doubleintegrator import doubleIntegratorTest
from pomp.example_problems.pendulum import pendulumTest
import pickle
import gym
from gym.wrappers.time_limit import TimeLimit
# from PendulumEnv import PendulumEnv
from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv, PendulumEnv
from wrappers import GoalWrapper

env_name = "HopperPyBulletEnv-v0"
# env_name = "Hopper-v2"
env_name = "Pendulum-v0"
RECORD_STEP=100
SAVE_STEP = 100000

def train_ppo():
  # env = Env(env_name)
  env = Env(gym.make(env_name))
  # env = Env(TimeLimit(PendulumEnv(g=9.8), max_episode_steps=200))
  # env = Env(TimeLimit(GoalWrapper(PendulumGoalEnv(g=9.8)), max_episode_steps=200))
  # env = Env(TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200))
  # env = Env(TimeLimit(PendulumEnv(g=9.8), max_episode_steps=200))

  # # problem = doubleIntegratorTest()
  # problem = pendulumTest()
  # # env = PlanningEnvGymWrapper(problem)
  # env._env = PendulumWrapper(problem, goal_conditioned=False)
  # env._env = KinomaticGymWrapper(problem, goal_conditioned=False)

  state_dim=env._env.reset().shape[0]
  action_dim=env._env.action_space.sample().shape[0]
  agent = ActorCritic(HIDDEN_SIZE, state_dim=state_dim, action_dim=action_dim)

  actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
  critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)

  rewards = []


  state, done, total_reward, D = env.reset(), False, 0, []
  pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
  ave_total_reward = 0
  for step in pbar:
    # Collect set of trajectories D by running policy π in the environment
    policy, value = agent(state)
    action = policy.sample()
    log_prob_action = policy.log_prob(action)
    next_state, reward, done = env.step(action)
    total_reward += reward
    D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'done': torch.tensor([done], dtype=torch.float32), 'log_prob_action': log_prob_action, 'old_log_prob_action': log_prob_action.detach(), 'value': value})
    state = next_state
    if step%RECORD_STEP == 0:
      rewards.append(ave_total_reward)

    if step%SAVE_STEP == 0:
      with open("saved_models/ppo_" + env_name + ".pkl", 'wb') as f:
        pickle.dump(agent, f)

    if done:
      pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
      plot(step, total_reward, 'ppo')
      ave_total_reward = ave_total_reward*.9  + total_reward*.1
      state, total_reward = env.reset(), 0

      if len(D) >= BATCH_SIZE:
        # Compute rewards-to-go R and advantage estimates based on the current value function V
        with torch.no_grad():
          reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
          for transition in reversed(D):
            reward_to_go = transition['reward'] + (1 - transition['done']) * (DISCOUNT * reward_to_go)
            transition['reward_to_go'] = reward_to_go
            td_error = transition['reward'] + (1 - transition['done']) * DISCOUNT * next_value - transition['value']
            advantage = td_error + (1 - transition['done']) * DISCOUNT * TRACE_DECAY * advantage
            transition['advantage'] = advantage
            next_value = transition['value']
        # Turn trajectories into a single batch for efficiency (valid for feedforward networks)
        trajectories = {k: torch.cat([trajectory[k] for trajectory in D], dim=0) for k in D[0].keys()}
        # Extra step: normalise advantages
        trajectories['advantage'] = (trajectories['advantage'] - trajectories['advantage'].mean()) / (trajectories['advantage'].std() + 1e-8)
        D = []

        for epoch in range(PPO_EPOCHS):
          # Recalculate outputs for subsequent iterations
          if epoch > 0:
            policy, trajectories['value'] = agent(trajectories['state'])
            trajectories['log_prob_action'] = policy.log_prob(trajectories['action'].detach())

          # Update the policy by maximising the PPO-Clip objective
          policy_ratio = (trajectories['log_prob_action'].sum(dim=1) - trajectories['old_log_prob_action'].sum(dim=1)).exp()
          policy_loss = -torch.min(policy_ratio * trajectories['advantage'], torch.clamp(policy_ratio, min=1 - PPO_CLIP_RATIO, max=1 + PPO_CLIP_RATIO) * trajectories['advantage']).mean()
          actor_optimiser.zero_grad()
          policy_loss.backward()
          actor_optimiser.step()

          # Fit value function by regression on mean-squared error
          value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
          critic_optimiser.zero_grad()
          value_loss.backward()
          critic_optimiser.step()

  # record_run(env_name, agent, algorithm_name="PPO-Clip")
  return rewards

def evaluate_ppo(label='PPO'):
  samples = 1
  rewards = []

  print(label)
  for i in range(samples):
    print(i)
    new_run_rewards = train_ppo()
    rewards.append(get_rolling_ave(new_run_rewards))

  return rewards

def evaluate():
  labels = ['PPO']
  rewards = [evaluate_ppo()]

  # pdb.set_trace()
  steps = [[MAX_STEPS*i/len(rewards[0][0]) for i in range(len(rewards[0][0]))] for _ in labels]#range(len(rewards[0]))]

  multiplot(steps, rewards, labels, 'PPO learning: ' +env_name)

# train_sopg(line_search=True, use_hessian=False, record=True)

evaluate()