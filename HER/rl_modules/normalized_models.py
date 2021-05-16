import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import numpy as np
from mpi4py import MPI
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = torch.zeros(self.size)
        self.local_sumsq = torch.zeros(self.size)
        self.local_count = torch.zeros(1)
        # get the total sum sumsq and sum count
        self.total_sum = torch.zeros(self.size)
        self.total_sumsq = torch.zeros(self.size)
        self.total_count = torch.zeros(1)
        # get the mean and std
        self.mean = torch.zeros(self.size)
        self.std = torch.ones(self.size)
        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (torch.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        # local_sum[...] = self._mpi_average(local_sum)
        # local_sumsq[...] = self._mpi_average(local_sumsq)
        # local_count[...] = self._mpi_average(local_count)
        local_sum[...] = torch.tensor(self._mpi_average(local_sum.numpy()))
        local_sumsq[...] = torch.tensor(self._mpi_average(local_sumsq.numpy()))
        local_count[...] = torch.tensor(self._mpi_average(local_count.numpy()))
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.clone().detach()
            local_sum = self.local_sum.clone().detach()
            local_sumsq = self.local_sumsq.clone().detach()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        # buf = torch.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        return v
        if clip_range is None:
            clip_range = self.default_clip_range
        return torch.clip((v - self.mean) / (self.std), -clip_range, clip_range)



# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        # self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'])
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        # x = self.norm1(x)
        x = F.selu(x)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class normalized_module(nn.Module):
    def _update_normalizer(self,  episode_batch, her_module, _preproc_og):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = _preproc_og(obs, g)
        # update
        inpt = torch.tensor(np.concatenate([transitions['obs'], transitions['g']], axis=-1))
        self.normalizer.update(inpt)
        self.normalizer.recompute_stats()
        # self.g_norm.update(transitions['g'])
        # recompute the stats
        # self.o_norm.recompute_stats()
        # self.g_norm.recompute_stats()


class actor(normalized_module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        self.normalizer = normalizer(env_params['obs'] + env_params['goal'])

    def forward(self, x):
        # x = self.normalizer.normalize(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions







class critic(normalized_module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.normalizer = normalizer(env_params['obs'] + env_params['goal'])

    def forward(self, x, actions):
        # x = self.normalizer.normalize(x)
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.selu(x)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
