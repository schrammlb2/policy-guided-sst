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
        # self.mu_layer.weight.data.fill_(0)
        # self.mu_layer.bias.data.fill_(0)
        self.log_std_layer = nn.Linear(256, env_params['action'])
        # self.log_std_layer.weight.data.fill_(0)
        # self.log_std_layer.bias.data.fill_(-1.)

    def forward(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
        # with_logprob = False
        x = self.norm1(x)
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        net_out = F.relu(self.fc3(x))


        mu = self.mu_layer(net_out)/100
        log_std = self.log_std_layer(net_out)/100 -1.
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

    # def normed_forward(self, obs, g, deterministic=False): 
    #     obs_norm = self.o_norm.normalize(obs)
    #     g_norm = self.g_norm.normalize(g)
    #     # concatenate the stuffs
    #     inputs = np.concatenate([obs_norm, g_norm])
    #     inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    #     return self.forward(inputs, deterministic=deterministic)

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
        return self.forward(inputs, deterministic=deterministic, forced_exploration=3)



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

class tdm_critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + env_params['goal'] + env_params['action'])
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, env_params['goal'])

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = self.norm1(x)
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        x = F.relu(self.fc3(x))
        q_value = (self.q_out(x)**2).sum(dim=-1)

        return q_value


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


class dual_critic(nn.Module):
    def __init__(self, env_params):
        super(dual_critic, self).__init__()
        self.q1 = critic(env_params)
        self.q2 = critic(env_params)

    def forward(self, x, actions):
        return torch.min(self.q1(x, actions), self.q2(x, actions))

    def dual(self, x, actions):
        return self.q1(x, actions), self.q2(x, actions)


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

    def norm(self, o: torch.Tensor, g: torch.Tensor):
        return self.actor._get_norms(o,g)
        
    def denorm(self, o: torch.Tensor, g: torch.Tensor):
        return self.actor._get_denorms(o,g)
    # def denorm_o(self, o: torch.Tensor):
    # def denorm_g(self, o: torch.Tensor):

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


# class SquashedGaussianMLPActor(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.act_limit = act_limit

#     def forward(self, obs, deterministic=False, with_logprob=True):
#         net_out = self.net(obs)
#         mu = self.mu_layer(net_out)
#         log_std = self.log_std_layer(net_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()

#         if with_logprob:
#             # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
#             # NOTE: The correction formula is a little bit magic. To get an understanding 
#             # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
#             # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
#             # Try deriving it yourself as a (very difficult) exercise. :)
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
#         else:
#             logp_pi = None

#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action

#         return pi_action#, logp_pi


# class MLPQFunction(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         c = torch.cat([obs,act], dim=-1)
#         #dev=torch.cuda.device_of(c)
#         q = self.q(c)
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.

# class MLPActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
#                  activation=nn.ReLU):
#         super().__init__()

#         obs_dim = observation_space.shape[0]
#         act_dim = action_space.shape[0]
#         act_limit = action_space.high[0]

#         # build policy and value functions
#         self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
#         self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
#         self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.pi(obs, deterministic, False)
#             return a.cpu().numpy()


# class state_critic(nn.Module):
#     def __init__(self, env_params):
#         super(state_critic, self).__init__()
#         # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.norm1 = nn.LayerNorm(256)
#         self.norm2 = nn.LayerNorm(256)
#         self.norm3 = nn.LayerNorm(256)
#         self.q_out = nn.Linear(256, 1)

#     def forward(self, x):
#         x = F.relu(self.norm1(self.fc1(x)))
#         x = F.relu(self.norm2(self.fc2(x)))
#         x = F.relu(self.norm3(self.fc3(x)))
#         q_value = self.q_out(x)

#         return q_value


# class net(nn.Module):
#     def __init__(self, input_shape, args):
#         super(net, self).__init__()
#         self.norm1 = nn.BatchNorm1d(input_shape)
#         # self.norm1 = nn.BatchNorm1d(args.hidden_size)
#         self.norm2 = nn.BatchNorm1d(args.hidden_size)
#         self.norm3 = nn.BatchNorm1d(args.hidden_size)
#         self.norm4 = nn.BatchNorm1d(args.hidden_size)
#         self.fc1 = nn.Linear(input_shape, args.hidden_size)
#         self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
#         self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
#         self.q_out = nn.Linear(args.hidden_size, 1)
#         self.do1 = nn.Dropout(args.dropout_rate)
#         self.do2 = nn.Dropout(args.dropout_rate)
#         self.do3 = nn.Dropout(args.dropout_rate)
#         self.act = F.relu

#     def forward(self, x):
#         x = self.norm1(x)
#         x = self.act(self.fc1(x))
#         x = self.norm2(x)
#         x = self.do1(x)
#         x = self.act(self.fc2(x))
#         x = self.norm3(x)
#         x = self.do2(x)
#         x = self.act(self.fc3(x))
#         x = self.norm4(x)
#         # x = self.do3(x)
#         q_value = self.q_out(x)

#         return q_value


