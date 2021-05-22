import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.action_out = nn.Linear(256, env_params['action'])

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions

# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.q_out = nn.Linear(256, 1)

#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         q_value = self.q_out(x)

#         return q_value


# class actor(nn.Module):
#     def __init__(self, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.norm1 = nn.BatchNorm1d(env_params['obs'] + env_params['goal'])
#         self.norm2 = nn.BatchNorm1d(256)
#         self.norm3 = nn.BatchNorm1d(256)
#         # self.norm1 = nn.LayerNorm(256)
#         # self.norm2 = nn.LayerNorm(256)
#         # self.norm3 = nn.LayerNorm(256)
#         self.action_out = nn.Linear(256, env_params['action'])
#         # self.action_out.weight.data.fill_(0)
#         # self.action_out.bias.data.fill_(0)
        

#     def forward(self, x):
#         x = F.relu(self.fc1(self.norm1(x)))
#         x = F.relu(self.fc2(self.norm2(x)))
#         x = F.relu(self.fc3(self.norm3(x)))
#         # x = F.relu(self.norm1(self.fc1(x)))
#         # x = F.relu(self.norm2(self.fc2(x)))
#         # x = F.relu(self.norm3(self.fc3(x)))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions



# class critic(nn.Module):
#     def __init__(self, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         # self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.norm1 = nn.BatchNorm1d(env_params['obs'] + env_params['goal']  + env_params['action'])
#         self.norm2 = nn.BatchNorm1d(256)
#         self.norm3 = nn.BatchNorm1d(256)
#         # self.norm1 = nn.LayerNorm(256)
#         # self.norm2 = nn.LayerNorm(256)
#         # self.norm3 = nn.LayerNorm(256)
#         self.q_out = nn.Linear(256, 1)
#         # self.q_out.weight.data.fill_(0)
#         # self.q_out.bias.data.fill_(0)

#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         # x = F.relu(self.norm1(self.fc1(x)))
#         # x = F.relu(self.norm2(self.fc2(x)))
#         # x = F.relu(self.norm3(self.fc3(x)))
#         x = F.relu(self.fc1(self.norm1(x)))
#         x = F.relu(self.fc2(self.norm2(x)))
#         x = F.relu(self.fc3(self.norm3(x)))
#         q_value = self.q_out(x)

#         return q_value





class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def normed_forward(self, obs, g, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs)


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value



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

    def forward(self, o: torch.Tensor, g: torch.Tensor): 
        assert type(o) == torch.Tensor
        assert type(g) == torch.Tensor
        obs_norm, g_norm = self.actor._get_norms(o,g)
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)

        action = self.actor(inputs)
        value = self.critic(inputs, action).squeeze()

        # return self.q2time(value)
        return value
