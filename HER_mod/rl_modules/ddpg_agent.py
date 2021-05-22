import torch
import os
from datetime import datetime
import numpy as np
import itertools
from mpi4py import MPI

from HER_mod.mpi_utils.mpi_utils import sync_networks, sync_grads
from HER_mod.mpi_utils.normalizer import normalizer

from HER_mod.her_modules.her import her_sampler

from HER_mod.rl_modules.replay_buffer import replay_buffer
from HER_mod.rl_modules.models import actor, critic
from HER_mod.rl_modules.value_map import *
from HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.hyperparams import POS_LIMIT

import pdb
"""
ddpg with HER (MPI-version)

"""
DTYPE = torch.float32

POLYAK_SCALE = .0
train_on_target = False
train_on_target = True


class ValueEstimator:
  def __init__(self, env_params, args):
    self.args = args

    self.double_q = False

    self.critic_1 = critic(env_params)
    self.critic_2 = critic(env_params)
    sync_networks(self.critic_1)
    sync_networks(self.critic_2)
    self.critic_target_1 = critic(env_params)
    self.critic_target_2 = critic(env_params)
    self.critic_target_1.load_state_dict(self.critic_1.state_dict())    
    self.critic_target_2.load_state_dict(self.critic_2.state_dict())    
    if self.double_q: 
        self.critics_optimiser = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.args.lr_critic, weight_decay=.004)
    else: 
        self.critics_optimiser = torch.optim.Adam(list(self.critic_1.parameters()))# + list(self.critic_2.parameters()), lr=self.args.lr_critic, weight_decay=.004)

    if self.args.cuda:
        self.critic_1.cuda()
        self.critic_2.cuda()
        self.critic_target_1.cuda()
        self.critic_target_2.cuda()

    self.polyak_base = self.args.polyak
    self.polyak_scale = POLYAK_SCALE
    self.polyak_decay = (1-.5/(5*self.args.n_cycles))
    self.scale = 1#0/ (1 - self.args.gamma)


  def min_critic(self, state, action):
    tc1 = self.critic_1(state, action)*self.scale
    if not self.double_q:
        return tc1
    tc2 = self.critic_2(state, action)
    return torch.min(tc1, tc2)

  def min_critic_target(self, state, action):
    tc1 = self.critic_target_1(state, action)*self.scale
    if not self.double_q:
        return tc1
    tc2 = self.critic_target_2(state, action)
    return torch.min(tc1, tc2)

  # def t_to_r(self, t):
  #   min_r = -1 / (1 - self.args.gamma)
  #   return min_r*(1-self.args.gamma**t)
    
  def q_loss(self, inputs_norm_tensor, inputs_next_norm_tensor, actor, transitions):
    with torch.no_grad():        
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        # do the normalization
        # concatenate the stuffs
        # actions_next = actor(inputs_next_norm_tensor)
        actions_next = actor(inputs_next_norm_tensor)
        shape = actions_next.shape
        TARGET_ACTION_NOISE = .1
        TARGET_ACTION_NOISE_CLIP = .25
        noise = torch.clamp(TARGET_ACTION_NOISE * torch.normal(torch.zeros(shape), torch.ones(shape)), min=-TARGET_ACTION_NOISE_CLIP, max=TARGET_ACTION_NOISE_CLIP)
        actions_next = torch.clamp(actions_next + noise, min=-1, max=1)
        # q_next_value = self.critic_target_network(critic_inputs_next_norm_tensor, actions_next)
        # q_next_value = self.min_critic_target(inputs_next_norm_tensor, actions_next)
        q_next_value = self.critic_target_1(inputs_next_norm_tensor, actions_next)
        # q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
        q_next_value = q_next_value.detach()

        target_q_value = r_tensor + self.args.gamma * q_next_value * (-r_tensor)# + 1
        target_q_value = target_q_value.detach()
        # clip the q value
        clip_return = 1 / (1 - self.args.gamma)
        # cum_r_val = -clip_return*(1-self.args.gamma**torch.tensor(2*transitions['t'], dtype=torch.float32) )

        target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)
        # top_clip = torch.min(target_q_value, torch.zeros_like(target_q_value))
        # target_q_value_clip = torch.max(top_clip, cum_r_val.unsqueeze(1))
        # import pdb
        # pdb.set_trace()
        # alpha = 0
        # target_q_value = (1-alpha)*target_q_value + alpha*target_q_value_clip


    if self.double_q:
        q1 = self.critic_1(inputs_norm_tensor, actions_tensor)
        q2 = self.critic_2(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - q1).pow(2).mean() + (target_q_value - q2).pow(2).mean()
    else:  
        q1 = self.critic_1(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - q1).pow(2).mean()
    return critic_loss


  # def _soft_update_target_network(self, target, source):
  #   for target_param, param in zip(target.parameters(), source.parameters()):
  #       target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
  def _soft_update_target_network(self, target, source):
    self.polyak_scale*=self.polyak_decay
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - (self.polyak_base- self.polyak_scale)) * param.data + (self.polyak_base- self.polyak_scale) * target_param.data)

  def update(self, inputs_norm_tensor, inputs_next_norm_tensor, actor, transitions):
    # Update Q-functions by one step of gradient descent
    self.critics_optimiser.zero_grad()
    self.q_loss(inputs_norm_tensor, inputs_next_norm_tensor, actor, transitions).backward()
    self.critics_optimiser.step()

    # Update target value network
    self._soft_update_target_network(self.critic_target_1, self.critic_1)
    if self.double_q: 
        self._soft_update_target_network(self.critic_target_2, self.critic_2)


class ddpg_agent:
    def __init__(self, args, env, env_params, vel_goal=True):
        self.global_count = 0
        self.goal_tuning = True
        self.replan = True
        # self.replan = False
        self.goal_tuning = False
        self.update_num=5
        self.gd_steps = 10

        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        # self.critic_network = critic(env_params)
        self.critic = ValueEstimator(env_params, args)
        self.planning_critic = self.critic#ValueEstimator(env_params, args)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        # sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        # self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        # self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            # self.critic_network.cuda()
            self.actor_target_network.cuda()
            # self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        # self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions, ddpg_sample_func=self.her_module.sample_ddpg_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        self.normalize = True
        self.vel_goal = vel_goal

        self.polyak_base = self.args.polyak
        self.polyak_scale = POLYAK_SCALE
        self.polyak_decay = (1-.5/(5*self.args.n_cycles))
        self.search_lr = .1

    def learn(self, hooks=[]):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                mb_col = []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    ep_col = []
                    # reset the environment
                    observation = self.env.reset()

                    random_next_goal = (np.random.rand(2)*2-1)*POS_LIMIT
                    if self.goal_tuning:
                        pos = observation['observation']
                        path = [self.env.goal, random_next_goal]
                        # time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
                        # gd_steps=np.random.geometric(p=.5)
                        time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
                        # time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps)
                        target_v = vel_path[0].detach().numpy() + np.random.standard_normal(2)*.2
                        self.env.set_new_vel_goal(target_v)
                        observation = self.env.get_state()


                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        # if 'collided' in observation_new.keys():
                        #     ep_col.append(observation_new['collided'])
                        ep_col.append(False)
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        # if done: 
                            # break
                            # observation = self.env.set_new_goal(random_next_goal)
                            # # observation = self.env.reset_goal()
                            # random_next_goal = (np.random.rand(2)*2-1)*POS_LIMIT
                            # if self.goal_tuning:
                            #     pos = observation['observation']
                            #     path = [self.env.goal, random_next_goal]
                            #     # time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
                            #     # gd_steps=np.random.geometric(p=.5)
                            #     time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
                            #     # time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps)
                            #     target_v = np.clip(vel_path[0].detach().numpy() + np.clip(np.random.standard_normal(2)*.4, -.8, .8), -1, 1)

                            #     self.env.set_new_vel_goal(target_v)
                            # observation = self.env.get_state()
                            # g = observation['desired_goal']

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_col.append(ep_col)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_col = np.array(mb_col)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_col])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_col])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                    # self._update_planning_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                # self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate, ave_reward  = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                # print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, average reward is {:.3f}'.format(datetime.now(), epoch, success_rate, ave_reward))
                [hook.run(self) for hook in hooks]
                # q_value_map(self.critic.min_critic, self.actor_network, title= "HER DDPG value map, epoch " + str(epoch))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        # inputs = np.concatenate([obs_norm, g_norm])
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        # pass
        mb_obs, mb_ag, mb_g, mb_actions, mb_col = episode_batch
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
                       'col': mb_col, 
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.actor_network.set_normalizers(self.o_norm.get_torch_normalizer(), self.g_norm.get_torch_normalizer())

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    # def _soft_update_target_network(self, target, source):
    #     for target_param, param in zip(target.parameters(), source.parameters()):
    #         target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)


    def _soft_update_target_network(self, target, source):
        self.polyak_scale*=self.polyak_decay
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - (self.polyak_base- self.polyak_scale)) * param.data + (self.polyak_base- self.polyak_scale) * target_param.data)


    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        # critic_inputs_norm = np.concatenate([obs_norm, g_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # critic_inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        self.global_count += 1
        # scale = 1/(1-self.args.gamma)
        if self.global_count % 2 == 0:
            actions_real = self.actor_network(inputs_norm_tensor)
            if train_on_target: 
                actor_loss = -self.critic.min_critic_target(inputs_norm_tensor, actions_real).mean()
            else: 
                actor_loss = -self.critic.min_critic(inputs_norm_tensor, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            # start to update the network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()
        # update the critic_network
        # self.critic.update(inputs_norm_tensor, inputs_next_norm_tensor, transitions, self.actor_target_network)
        self.critic.update(inputs_norm_tensor, inputs_next_norm_tensor,  self.actor_target_network, transitions)


    def _update_planning_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size, off_goal=False)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        # critic_inputs_norm = np.concatenate([obs_norm, g_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # critic_inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()


        self.planning_critic.update(inputs_norm_tensor, inputs_next_norm_tensor,  self.actor_target_network, transitions)
    # do the evaluation

    def _eval_agent(self, verbose=False):
        total_success_rate = []
        total_reward = []
        run_num = self.args.n_test_rollouts
        if verbose: 
            run_num = 1
        for _ in range(run_num):
            # per_success_rate = []
            total_r = 0
            success = 0
            observation = self.env.reset()

            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    # input_tensor = self._preproc_inputs(obs, g)
                    # pi = self.actor_network(input_tensor)
                    pi = self.actor_network.normed_forward(obs, g)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze(axis=0)
                observation_new, r, done, info = self.env.step(actions)
                total_r += r
                if verbose: 
                    print(observation_new)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                # per_success_rate.append(info['is_success'])
                success = info['is_success']
                if done: 
                    break


            # total_success_rate.append(per_success_rate)
            total_success_rate.append(success)
            total_reward.append(total_r)
        total_success_rate = np.array(total_success_rate)
        total_reward = np.array(total_reward)
        # local_success_rate = np.mean(total_success_rate[:, -1])
        local_success_rate = np.mean(total_success_rate)
        local_reward = np.mean(total_reward)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_reward / MPI.COMM_WORLD.Get_size()


    def get_actions(self, observations, goals):
        obs_norm = self.o_norm.normalize(observations)
        g_norm = self.g_norm.normalize(goals)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        actions = self.actor_network(torch.tensor(inputs_norm, dtype=torch.float32)).detach().numpy()
        return actions

    # def q2time(self, q):
    #     max_q = 1/(1-self.args.gamma)
    #     ratio = -.99*torch.clip(q/max_q, -1,0) #.99 for numerical stability
    #     return torch.log(1-ratio)/torch.log(torch.tensor(self.args.gamma))

    def q2time(self, q):
        # max_q = 1/(1-self.args.gamma)
        # ratio = -.99*torch.clip(q/max_q, -1,0) #.99 for numerical stability
        return torch.log(1+q*(1-self.args.gamma)*.99)/torch.log(torch.tensor(self.args.gamma))

    def time2q(self, time):
        return (1-self.args.gamma**time)/(1-self.args.gamma)


    def _eval_path_q_network(self, pos, loc_path, vel_path):
        if self.vel_goal:
            path = [torch.cat([loc, vel], dim=-1) for loc, vel, in zip(loc_path, vel_path)]
        else: 
            path = loc_path
        if pos.shape[-1] < 4:
            prev_loc = np.concatenate([pos, np.zeros(2)]) #Start with 0 velocity
        else: 
            prev_loc = pos
        prev_loc = torch.tensor(prev_loc, dtype=DTYPE)
        time = 0
        q = 0
        for goal in path:
            if self.normalize:
                o = self.o_norm.torch_normalize(prev_loc)
                g = self.g_norm.torch_normalize(goal)
            else: 
                o, g = prev_loc, goal
            # input_tensor = torch.cat([prev_loc, goal], dim=-1).unsqueeze(0)
            input_tensor = torch.cat([o, g], dim=-1).unsqueeze(0)
            action = self.actor_network(input_tensor)
            # time += self.critic.min_critic(input_tensor, action)
            # time += self.critic.critic_1(input_tensor, action)
            # time += self.q2time(self.critic.critic_1(input_tensor, action))
            new_q_val = -self.planning_critic.min_critic_target(input_tensor, action)
            time += self.q2time(new_q_val).detach()
            q = q + new_q_val*self.args.gamma**time
            # time += self.critic.min_critic(input_tensor, action)

            if self.vel_goal:
                prev_loc = goal
            else: 
                prev_loc = np.concatenate([goal, np.zeros(2)])#goal
                prev_loc = torch.tensor(prev_loc, dtype=DTYPE)
        # return -time
        return q



    def run_path(self, initial_pos, loc_path, vel_path, random_search=False):
        env = MultiGoalEnvironment("asdf", time=True, vel_goal=False)
        # obs = initial_pos
        env.set_state(initial_pos.numpy()[:2], np.zeros(2))
        obs = env.get_state()['observation']
        time = 0
        successful = True
        trajectory = []
        hit_points = []
        goals = []
        step_num=0
        for i in range(len(loc_path)): 
            g = loc_path[i]
            v = vel_path[i]
            done = False
            env.set_new_goal(g.numpy())
            state = env.get_state()
            while not done: 
                obs = torch.tensor(obs, dtype=DTYPE)
                step_num += 1
                if self.replan and step_num % self.update_num == 0:
                    # if i > 0:
                    #     import pdb
                    #     pdb.set_trace()
                    if random_search:
                        _, vel_path[i:] = self.evaluate_path_random_search(obs, loc_path[i:], gd_steps=self.gd_steps)#, vel_path=vel_path)
                    else: 
                        _, vel_path[i:] = self.evaluate_path(obs, loc_path[i:], gd_steps=self.gd_steps)
                    #g and v should already be normalized
                if self.normalize:
                    o = self.o_norm.torch_normalize(obs)
                    if self.vel_goal:
                        goal = self.g_norm.torch_normalize(torch.cat([g, v], dim=-1))
                    else: 
                        goal = self.g_norm.torch_normalize(g)
                else: 
                    o, goal = prev_loc, torch.cat([g, v], dim=-1)
                # input_tensor = torch.cat([prev_loc, goal], dim=-1).unsqueeze(0)
                with torch.no_grad():
                    input_tensor = torch.cat([o, goal], dim=-1).unsqueeze(0)
                    # input_tensor = torch.cat([obs, g, v], dim=-1)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, done, info = env.step(actions)
                time += 1#reward

                obs = observation_new['observation']
                if observation_new['collided']: 
                    time += 50
                    # done = True
                    # info['is_success'] = False

                trajectory.append(obs[:2])
                if done: 
                    hit_points.append(obs[:2])
                    goals.append(observation_new['desired_goal'])
                # per_success_rate.append(info['is_success'])
            if not info['is_success']: 
                successful = False
                break

        return time, trajectory, successful, (hit_points, goals)


    def norm_clip(self, vec, max_mag):
        norm = torch.norm(vec*1.0)
        if norm > max_mag:
            return vec*(max_mag/norm.detach())
        else: 
            return vec



    def evaluate_path(self, pos, path, gd_steps=0, random_start=False, vel_path="", last_static=False):
        # norm_path = [torch.tensor(self.g_norm.normalize(p)) for p in path]
        loc_path = [torch.tensor(p, dtype=DTYPE) for p in path] #Not doing the normalizing rn bc it's too much hassle
        if vel_path == "":
            if random_start: 
                vel_path = [torch.tensor(.8*(torch.rand(2)*2-1), requires_grad=True) for p in path]
            else:
                vel_path = [torch.zeros(2, requires_grad=True) for p in path]
        # pos = torch.tensor(self.o_norm(pos))]
        pos = torch.tensor(pos)
        if gd_steps > 0:
            if last_static: 
                velocity_opt = torch.optim.Adam(vel_path[:-1], lr=self.search_lr)#self.args.lr_actor)
            else:
                velocity_opt = torch.optim.Adam(vel_path, lr=self.search_lr)
            # velocity_opt = torch.optim.SGD(vel_path, lr=.05)
            for i in range(gd_steps):
                r_max = .8
                # [torch.clamp(p, min=-r_max, max=r_max) for p in vel_path]
                for i in range(len(vel_path)):
                    norm = torch.norm(vel_path[i]).detach()
                    # if norm > r_max: 
                    #     vel_path[i].data *= (r_max/norm.detach())
                # vel_path = [self.norm_clip(v, .8) for v in vel_path]
                time = self._eval_path_q_network(pos, loc_path, vel_path)
                velocity_opt.zero_grad()
                time.backward()
                # import pdb
                # pdb.set_trace()
                velocity_opt.step()
                with torch.no_grad():
                    [p.clamp_(min=-r_max, max=r_max) for p in vel_path]
        else:
            time = self._eval_path_q_network(pos, loc_path, vel_path)

        # time, trajectory, successful, pass_vals = self.run_path(pos, loc_path, vel_path)
        # return time, trajectory, vel_path, successful, pass_vals
        return time, vel_path#, trajectory, vel_path, successful, pass_vals

    def evaluate_path_random_search(self, pos, path, gd_steps=0):
        loc_path = [torch.tensor(p, dtype=DTYPE) for p in path] #Not doing the normalizing rn bc it's too much hassle
        pos = torch.tensor(pos)
        min_time = float("inf")
        min_vel_path = [torch.zeros(2, dtype=DTYPE) for p in path]
        time = self._eval_path_q_network(pos, loc_path, min_vel_path)

        for i in range(gd_steps*2):
            # [torch.clamp(p, min=-.8, max=.8) for p in vel_path]
            vel_path = [.8*(torch.rand(2)*2-1) for p in path]
            time = self._eval_path_q_network(pos, loc_path, vel_path)
            if time < min_time:
                min_vel_path = vel_path
                min_time = time

        vel_path = min_vel_path
        time = self._eval_path_q_network(pos, loc_path, vel_path)
        return time, vel_path#, trajectory, vel_path, successful, pass_vals


    def find_shortest_path(self, pos, goals, gd_steps=0, random_search=False, random_start=False, perm_search=True):
        min_time = float("inf")
        min_path = None
        min_vel_path = None
        found_success = False
        last_pass_vals = None
        old_gd_steps = self.gd_steps
        self.gd_steps = gd_steps
        # min_trajectory
        if perm_search:
            for path in itertools.permutations(goals):
                if random_search:
                    time, vel_path = self.evaluate_path_random_search(pos, path, gd_steps=gd_steps)
                else:
                    time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps, random_start=random_start)

                if time < min_time:# and successful: 
                    min_time = time
                    min_path = path
                    min_vel_path = vel_path
        else: 
            path = goals
            if random_search:
                time, vel_path = self.evaluate_path_random_search(pos, path, gd_steps=gd_steps)
            else:
                time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps, random_start=random_start)

            if time < min_time:# and successful: 
                min_time = time
                min_path = path
                min_vel_path = vel_path


        loc_path = [torch.tensor(p, dtype=DTYPE) for p in min_path]
        time, min_trajectory, successful, last_pass_vals = self.run_path(torch.tensor(pos), loc_path, vel_path, random_search=random_search)
        self.gd_steps = old_gd_steps
        found_success = successful
        return (min_time, min_trajectory, min_path, min_vel_path,  found_success, last_pass_vals)


    def find_shortest_path(self, pos, goals, gd_steps=0, random_search=False, random_start=False, perm_search=True):
        min_time = float("inf")
        min_path = None
        min_vel_path = None
        found_success = False
        last_pass_vals = None
        old_gd_steps = self.gd_steps
        self.gd_steps = gd_steps
        # min_trajectory
        if perm_search:
            for path in itertools.permutations(goals):
                if random_search:
                    time, vel_path = self.evaluate_path_random_search(pos, path, gd_steps=gd_steps)
                else:
                    time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps, random_start=random_start)

                if time < min_time:# and successful: 
                    min_time = time
                    min_path = path
                    min_vel_path = vel_path
        else: 
            path = goals
            if random_search:
                time, vel_path = self.evaluate_path_random_search(pos, path, gd_steps=gd_steps)
            else:
                time, vel_path = self.evaluate_path(pos, path, gd_steps=gd_steps, random_start=random_start)

            if time < min_time:# and successful: 
                min_time = time
                min_path = path
                min_vel_path = vel_path


        loc_path = [torch.tensor(p, dtype=DTYPE) for p in min_path]
        time, min_trajectory, successful, last_pass_vals = self.run_path(torch.tensor(pos), loc_path, vel_path, random_search=random_search)
        self.gd_steps = old_gd_steps
        found_success = successful
        return (min_time, min_trajectory, min_path, min_vel_path,  found_success, last_pass_vals)



    def select_path(self, pos, goals, method=""):
        min_time = float("inf")
        min_path = None
        min_vel_path = None
        found_success = False
        last_pass_vals = None
        old_gd_steps = self.gd_steps
        self.gd_steps = 10
        # ten_gd_steps = 15
        # min_trajectory
        path = goals
        self.replan = False
        if method == "random search":
            time, vel_path = self.evaluate_path_random_search(pos, path, gd_steps=self.gd_steps)
        elif method == "gradient descent":
            time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
        elif method == "gradient descent (40 steps)":
            self.gd_steps = 40
            time, vel_path = self.evaluate_path(pos, path, gd_steps=self.gd_steps)
        elif method == "random": 
            time, vel_path = 0, [torch.rand(2, dtype=DTYPE) for _ in goals]
        elif method == "0 velocity target": 
            time, vel_path = 0, [torch.zeros(2, dtype=DTYPE) for _ in goals]

        if "replanning" in method: 
            self.replanning = True


        time, min_trajectory, successful, last_pass_vals = self.run_path(torch.tensor(pos), torch.tensor(goals, dtype=DTYPE), vel_path)

        # if method == "gradient descent":
        #     import pdb
        #     pdb.set_trace()

        # print("\n-------------------------------------")
        # print("Method: " + method)
        # print("Vel path: " + str(vel_path))
        # print("Time: " + str(len(min_trajectory)))
        # len(min_trajectory)
        # import pdb
        # pdb.set_trace()
        
        self.gd_steps = old_gd_steps
        found_success = successful
        return (min_time, min_trajectory, min_path, min_vel_path,  found_success, last_pass_vals)

