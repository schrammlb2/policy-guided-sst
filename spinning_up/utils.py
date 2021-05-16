import os
import matplotlib.pyplot as plt
from spinning_up.hyperparams import MAX_STEPS
import pdb
import numpy as np

class Record:
  def __init__(self):
    self.steps, self.rewards = [], []
    self.runs = []

  def reset(self):
    self.steps, self.rewards = [], []

  def save_run(self):
    runs.append(np.array(rewards))

rec = Record()
var_steps, variances = [], []

def mean(lst):
  return sum(lst)/len(lst)

def std(lst):
  mn = mean(lst)
  return sum([(elem - mn)**2 for elem in lst])**.5

ave_scale = .95

def get_rolling_ave(lst):
  sum_val = 0
  div = 0
  ave_list = []
  for elem in lst: 
    sum_val = ave_scale*sum_val + (1-ave_scale)*elem
    div = ave_scale*div + (1-ave_scale)
    new_val = sum_val/div
    ave_list.append(new_val)

  return ave_list

def plot(step, reward, title):
  rec.steps.append(step)
  rec.rewards.append(reward)
  plt.plot(rec.steps, get_rolling_ave(rec.rewards), 'b-')
  # plt.plot(rec.steps, rec.rewards, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.xlim((0, MAX_STEPS))
  # plt.ylim((-2000, 1000))
  plt.savefig(os.path.join('spinning_up/results', title + '.png'))
  plt.close()

def plot_variance(step, variance, title):
  var_steps.append(step)
  variances.append(variance)
  # import pdb
  # pdb.set_trace()
  plt.plot(var_steps, variances, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('variance')
  plt.xlim((0, MAX_STEPS))
  plt.savefig(os.path.join('results', title + '_variance.png'))
  plt.close()

# def plot_with_error_bars(steps, rewards, title, label='Rewards', finished=True, color = 'g'):
#   samples = len(rewards)
#   rewards = np.array(rewards)
#   means = np.mean(rewards, axis=0)
#   stds = np.std(rewards, axis=0)
#   steps = np.array(steps)
#   conf_ints = 2*stds*samples**(-1/2)

#   # pdb.set_trace()
#   plt.plot(np.array(steps), np.array(means), color=color, label=label)
#   plt.fill_between(np.array(steps), np.array(means+conf_ints), np.array(means-conf_ints), color=color, alpha=.7)
  # plt.xlabel('Steps')
  # plt.ylabel('Rewards')
  # min_steps = steps.min()
  # max_steps = steps.max()
  # plt.xlim((min_steps, max_steps))
  # min_reward = (means-conf_ints*1.5).min()
  # max_reward = (means+conf_ints*1.5).max()
  # plt.ylim((min_reward, max_reward))
  # if finished:
    

# def multiplot(steps_list, rewards_list, label_list, title):
#   finished = False
#   color_cycle = ['b', 'r', 'c', 'g']
#   plt.figure(1)
#   plt.title(title)
#   for i in range(len(rewards_list)):
#     if i == len(rewards_list)-1:
#       finished = True
#       # break
#     plot_with_error_bars(steps_list[i], rewards_list[i], title, label_list[i], finished=finished, color=color_cycle[i])
#   plt.legend()
#   plt.savefig(os.path.join('results', title + '.png'))
#   plt.close() 


def multiplot(steps_list, rewards_list, label_list, title):
  finished = False
  color_cycle = ['b', 'r', 'c', 'g']
  plt.figure(1)
  plt.title(title)
  rewards_list = np.array(rewards_list)
  samples = rewards_list.shape[1]
  # import pdb
  # pdb.set_trace()

  means = np.mean(rewards_list, axis=1)
  stds = np.std(rewards_list, axis=1)
  steps_list = np.array(steps_list)
  conf_ints = 2*stds*samples**(-1/2)

  
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  # min_steps = steps.min()
  # max_steps = steps.max()
  # plt.xlim((min_steps, max_steps))
  # min_reward = (means-conf_ints*1.5).min()
  # max_reward = (means+conf_ints*1.5).max()
  # plt.ylim((min_reward, max_reward))
  for i in range(len(rewards_list)):
    color=color_cycle[i]
    plt.plot(steps_list[i], means[i], color=color, label=label_list[i])
    plt.fill_between(steps_list[i], np.array(means+conf_ints)[i], np.array(means-conf_ints)[i], color=color, alpha=.4)
    # plot_with_error_bars(steps_list[i], rewards_list[i], title, label_list[i], finished=finished, color=color_cycle[i])
  plt.legend()
  plt.savefig(os.path.join('results', title + '.png'))
  plt.close() 
