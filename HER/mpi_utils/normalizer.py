import threading
import numpy as np
import torch
from mpi4py import MPI

class torch_normalizer:
    def __init__(self, means, stds, clip_range): 
        self.mean = means
        self.std = stds
        self.mean = torch.tensor(means)
        self.std = torch.tensor(stds)
        self.clip_range = clip_range

    def normalize(self, v): 
        clip_range = self.clip_range
        if clip_range is None:
            clip_range = self.default_clip_range
        return torch.clip((v - self.mean) / (self.std), -clip_range, clip_range)
        # return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
    def denormalize(self, v):
        return v*self.std + self.mean


class normalizer:
    def __init__(self, size, eps=1e-8, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        # v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
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
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        # return v
        if clip_range is None:
            clip_range = self.default_clip_range
        normed = (v - self.mean) / (self.std)
        clip_val = np.clip(normed, -clip_range, clip_range)
        # if ((normed-clip_val)**2).sum() > .0001:
        #     import pdb
        #     pdb.set_trace()
        return clip_val


    def denormalize(self, v):
        return v*self.std + self.mean

    def get_torch_normalizer(self): 
        return torch_normalizer(self.mean, self.std, self.default_clip_range)



gamma = .95

class variant_normalizer(normalizer):
    def __init__(self, size, eps=1e-8, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_var = np.zeros(self.size, np.float32)
        self.local_mean = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_var = np.zeros(self.size, np.float32)
        self.total_mean = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    

    def update(self, v):
        with self.lock:
            self.local_var += v.var(axis=0)
            self.local_mean += v.mean(axis=0)
            self.local_count += 1


    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count
            local_mean = self.local_mean.copy()
            local_var = self.local_var.copy()
            # reset
            self.local_count = np.zeros(1)
            self.local_mean[...] = 0
            self.local_var[...] = 0
        # synrc the stats
        sync_mean, sync_var, sync_count = self.sync(local_mean, local_var, local_count)
        # update the total stuff
        self.total_mean += sync_mean
        self.total_var += sync_var
        self.total_count += sync_count
        # calculate the new mean and std

        self.mean = self.total_mean / self.total_count
        self.std  = (self.total_var / self.total_count)**.5

    