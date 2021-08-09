import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

#CUDA = torch.cuda.is_available()
#if CUDA:  
#    gpu_count = torch.cuda.device_count()
#    import random
#    DEVICE = torch.device(random.randint(gpu_count)) #Randomly assign to one of the GPUs
#else:
#    DEVICE = torch.device('cpu')












LOG_STD_MAX = 2
LOG_STD_MIN = -20

clip_max = 3



class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'])
        # self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        self.log_std_layer = nn.Linear(256, env_params['action'])

    def forward(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
        # with_logprob = False
        x = self.norm1(x)
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        net_out = F.relu(self.fc3(x))


        mu = self.mu_layer(net_out)#/100
        log_std = self.log_std_layer(net_out)-1#/100 -1.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)*forced_exploration

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action * pi_action

        if with_logprob: 
            return pi_action, logp_pi
        else: 
            return pi_action
        # return actions

    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs, deterministic=deterministic, forced_exploration=1)



class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'] + env_params['action'])
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = self.norm1(x)
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class exploration_actor(nn.Module):
    def __init__(self, env_params, rff_map):
        super(exploration_actor, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'])
        # self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc_rff = nn.Linear(2*env_params['rff_features'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        self.log_std_layer = nn.Linear(256, env_params['action'])
        self.rff_map = rff_map

    def forward(self, x, rff_state=None, rff_visit_states=None,
        with_logprob = False, deterministic = False, forced_exploration=1):
        if type(rff_state) == type(None): 
            rff_state = torch.zeros(x.shape[:-1] + (self.env_params['rff_features'],))
        if type(rff_visit_states) == type(None): 
            rff_visit_states = torch.zeros(x.shape[:-1] + (self.env_params['rff_features'],))

        # rff_state = rff_state*0
        # rff_visit_states = rff_visit_states*0
        x = self.norm1(x)
        # x = torch.cat([x, rff_state, rff_visit_states], dim=-1)
            #Explicitly do the norm *before* adding rrf
            #Norming the RRF would make them large and favor them too much in the regression
        # with_logprob = False
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x) + self.fc_rff(torch.cat([rff_state, rff_visit_states], dim=-1)))
        # x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        net_out = F.relu(self.fc3(x))


        mu = self.mu_layer(net_out)#/100
        log_std = self.log_std_layer(net_out)-1#/100 -1.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)*forced_exploration

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action * pi_action

        if with_logprob: 
            return pi_action, logp_pi
        else: 
            return pi_action
        # return actions

    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, rff_state=None, rff_visit_states=None, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs, rff_state, rff_visit_states, deterministic=deterministic, forced_exploration=1)



class exploration_critic(nn.Module):
    def __init__(self, env_params):
        super(exploration_critic, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'] + env_params['action'])
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        rff_encoding_dim = 3*env_params['rff_features'] + 1
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] 
        #     + rff_encoding_dim , 256)
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'] , 256)
        self.fc_rff = nn.Linear(rff_encoding_dim , 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.rff_re_add = nn.Linear(rff_encoding_dim, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions, rff_state=None, rff_visit_states=None):
        # if type(rff_state) == type(None): 
        #     rff_state = torch.zeros(x.shape[:-1] + (self.env_params['rff_features'],))
        # if type(rff_visit_states) == type(None): 
        #     rff_visit_states = torch.zeros(x.shape[:-1] + (self.env_params['rff_features'],))
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = self.norm1(x)
        prod = rff_state*rff_visit_states
        rff = torch.cat([rff_state, rff_visit_states, prod, torch.unsqueeze(torch.sum(prod, dim=-1), dim=-1)], dim=1)
        # x = torch.cat([x, rrf], dim=1)
        x = torch.clip(x, -clip_max, clip_max)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x) + self.fc_rff(rff))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        x = F.relu(self.fc3(x) + self.rff_re_add(rff)*.25)
        q_value = self.q_out(x)

        return q_value





def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class dual_critic(nn.Module):
    def __init__(self, env_params):
        super(dual_critic, self).__init__()
        self.q1 = critic(env_params)
        self.q2 = critic(env_params)

    def forward(self, x, actions):
        return torch.min(self.q1(x, actions), self.q2(x, actions))

    def dual(self, x, actions):
        return self.q1(x, actions), self.q2(x, actions)


class StateValueEstimator(nn.Module):
    def __init__(self, actor, critic, gamma):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def q2time(self, q):
        # max_q = 1/(1-self.args.gamma)
        # ratio = -.99*torch.clip(q/max_q, -1,0) #.99 for numerical stability
        return torch.log(1+q*(1-self.gamma)*.998)/torch.log(torch.tensor(self.gamma))

    def forward(self, o: torch.Tensor, g: torch.Tensor, norm=True): 
        assert type(o) == torch.Tensor
        assert type(g) == torch.Tensor
        if norm: 
            obs_norm, g_norm = self.actor._get_norms(o,g)
        else: 
            obs_norm, g_norm = o, g
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)

        action = self.actor(inputs)
        value = self.critic(inputs, action).squeeze()

        # return self.q2time(value)
        return value

