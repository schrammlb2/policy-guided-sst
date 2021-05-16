import copy
import torch
from torch import nn
from torch.distributions import Distribution, Normal
from spinning_up.hyperparams import ACTION_DISCRETISATION, DISCOUNT


class Actor(nn.Module):
  def __init__(self, hidden_size, stochastic=True, layer_norm=False, state_dim=3, action_dim=1):
    super().__init__()
    layers = [nn.Linear(state_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_dim)]
    layers[-1].weight.data.fill_(0)
    layers[-1].bias.data.fill_(0)
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.policy = nn.Sequential(*layers)
    if stochastic:
      self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))
      # self.policy_log_std = nn.Parameter(torch.tensor([[2.]]))

  def forward(self, state):
    policy = self.policy(state)
    return policy


class TanhNormal(Distribution):
  def __init__(self, loc, scale):
    super().__init__()
    self.normal = Normal(loc, scale)

  def sample(self):
    return torch.tanh(self.normal.sample())

  def rsample(self):
    return torch.tanh(self.normal.rsample())

  # Calculates log probability of value using the change-of-variables technique (uses log1p = log(1 + x) for extra numerical stability)
  def log_prob(self, value):
    inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
    return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6)  # log p(f^-1(y)) + log |det(J(f^-1(y)))|

  @property
  def mean(self):
    return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):
  def __init__(self, hidden_size, state_dim=3, action_dim=1):
    super().__init__()
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    layers = [nn.Linear(state_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2*action_dim)]
    self.policy = nn.Sequential(*layers)

  def forward(self, state):
    policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    policy = TanhNormal(policy_mean, policy_log_std.exp())
    return policy


class Critic(nn.Module):
  def __init__(self, hidden_size, state_action=False, layer_norm=False, state_dim=3, action_dim=1):
    super().__init__()
    self.state_action = state_action
    layers = [nn.Linear(state_dim + (action_dim if state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]

    layers[-1].weight.data.fill_(0)
    layers[-1].bias.data.fill_(0)
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.value = nn.Sequential(*layers)

  def forward(self, state, action=None):
    if self.state_action:
      value = self.value(torch.cat([state, action], dim=-1))
    else:
      value = self.value(state)
    return value.squeeze(dim=1)


class TimeCritic(nn.Module):
  def __init__(self, hidden_size, state_action=False, layer_norm=False, state_dim=3, action_dim=1):
    super().__init__()
    self.state_action = state_action
    layers = [nn.Linear(state_dim + (action_dim if state_action else 0) + 1, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.value = nn.Sequential(*layers)

  def forward(self, state, action,time):
    # scale_factor = time
    scale_factor = (1-DISCOUNT**time).float()#/(1-DISCOUNT)
    if self.state_action:
      # value = self.value(torch.cat([state, action, time], dim=1))*scale_factor
      value = self.value(torch.cat([state, action, scale_factor], dim=1))*scale_factor
    else:
      value = self.value(state)*scale_factor
    return value.squeeze(dim=1)


class ActorCritic(nn.Module):
  def __init__(self, hidden_size, state_dim=3, action_dim=1):
    super().__init__()
    self.actor = Actor(hidden_size, stochastic=True, state_dim=state_dim, action_dim=action_dim)
    self.critic = Critic(hidden_size, state_dim=state_dim, action_dim=action_dim)

  def forward(self, state):
    policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
    value = self.critic(state)
    return policy, value


class DQN(nn.Module):
  def __init__(self, hidden_size, num_actions=5, state_dim=3, action_dim=1):
    #Not sure how to interface action_dim with 'num_actions'. Saving this for later
    super().__init__()
    layers = [nn.Linear(state_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, num_actions*action_dim)]
    self.dqn = nn.Sequential(*layers)
    self.action_dim = action_dim

  def forward(self, state):
    values = self.dqn(state).view(-1, self.action_dim, ACTION_DISCRETISATION)
    return values

class TimeDQN(nn.Module):
  def __init__(self, hidden_size, num_actions=5, state_dim=3, action_dim=1):
    #Not sure how to interface action_dim with 'num_actions'. Saving this for later
    super().__init__()
    layers = [nn.Linear(state_dim+1, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, num_actions*action_dim)]
    self.dqn = nn.Sequential(*layers)

  def forward(self, state, time):
    values = self.dqn(torch.cat([state, time], dim=-1)).view(-1, self.action_dim, ACTION_DISCRETISATION)
    return values

def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_factor):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data
