import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from HER.mpi_utils.mpi_utils import sync_networks, sync_grads
from HER.mpi_utils.normalizer import normalizer
from HER.rl_modules.replay_buffer import replay_buffer
from HER.rl_modules.sac_models import actor, critic, dual_critic
from HER.her_modules.her import her_sampler
import copy
from sample_valid_goal import sample_valid_goal

"""
ddpg with HER (MPI-version)

"""
# critic_constructor = critic
critic_constructor = dual_critic

def reset_sim(env, mean, std):
    config = np.random.uniform(mean + std/20, mean - std/20)
    padded_config = np.concatenate([np.zeros(0), config], axis=-1)
    # self.env.sim.set_state_from_flattened(config)
    env.sim.set_state_from_flattened(padded_config)

    # Randomize start position of object.
    # if self.env.has_object:
    #     object_xpos = self.env.initial_gripper_xpos[:2]
    #     while np.linalg.norm(object_xpos - self.env.initial_gripper_xpos[:2]) < 0.1:
    #         object_xpos = self.env.initial_gripper_xpos[:2] + self.np_random.uniform(-self.env.obj_range, self.env.obj_range, size=2)
    #     object_qpos = self.env.sim.data.get_joint_qpos('object0:joint')
    #     assert object_qpos.shape == (7,)
    #     object_qpos[:2] = object_xpos
    #     self.env.sim.data.set_joint_qpos('object0:joint', object_qpos)

    env.sim.forward()
    return True


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.observation_space = self.env.observation_space['observation']
        env_params['goal'] = env_params['obs']
        env_params['max_timesteps'] = 5
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        # self.critic_network = critic(env_params)
        self.critic_network = critic_constructor(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        # self.critic_target_network = critic(env_params)
        self.critic_target_network = critic_constructor(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # self.reward_fn = lambda o, g, x=None: (np.mean((o-g)**2, axis=-1) < .0025) -1
        scale = .25
        self.sparse_reward_fn = lambda o, g, x=None: (np.mean(np.abs(o-g), axis=-1)/scale < 1) -1
        # self.sparse_reward_fn = lambda o, g, x=None: (np.all(np.abs(o-g)/scale < 1, axis=-1)) -1
        # self.sparse_reward_fn = lambda o, g, x=None: (np.sum(np.abs(o-g), axis=-1)/(scale*o.shape[0]) < 1) -1
        self.dense_reward_fn = lambda o, g, x=None: -np.mean((o-g)**2, axis=-1)**.5*50
        # self.dense_reward_fn = lambda o, g, x=None: -np.mean(
        #     (self.o_norm.normalize(o)-self.g_norm.normalize(g))**2
        #     , axis=-1)**.5
        # self.sparse_reward_fn = lambda o, g, x=None: (-self.dense_reward_fn(o, g) < scale) - 1

        # her sampler
        # self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.dense_reward_fn)
        # self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.sparse_reward_fn)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        self.reward_threshhold = -.5
        self.env_steps = 3
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def sample_goal(self):
        if np.random.binomial(1, .5):
            normed_goal = np.random.randn(self.env_params['goal'])
            goal = self.g_norm.denormalize(normed_goal)
            return goal
        else: 
            return sample_valid_goal(self.env)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    # g = self.observation_space.sample()



                    # if epoch > 1: 
                    #     # reset_sim(self.env, self.o_norm.mean, self.o_norm.std)
                    #     reset_sim(self.env, self.o_norm.mean, self.o_norm.std)
                    #     observation = self.env.env._get_obs()

                    obs = observation['observation']
                    ag =  observation['observation']
                    g = self.sample_goal()

                    # if epoch < 1: 
                    #     g = sample_valid_goal(self.env)
                    # else:
                    #     g = self.sample_random_goal()
                    # g = self.sample_random_goal()

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        # ag_new = observation_new['achieved_goal']
                        ag_new = observation_new['observation']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

    def sample_random_goal(self): 
        sample = np.random.randn(self.env_params['goal'])
        return self.g_norm.denormalize(sample)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
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
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

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
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
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
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next, log_prob_next = self.actor_target_network(inputs_next_norm_tensor, with_logprob = True)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next) + self.args.entropy_regularization*log_prob_next 
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value * (r_tensor < self.reward_threshhold) 
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        # real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        # critic_loss = (target_q_value - real_q_value).pow(2).mean()
        real_q_1, real_q_2 = self.critic_network.dual(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_1).pow(2).mean() + (target_q_value - real_q_2).pow(2).mean()
        # the actor loss
        actions_real, log_prob = self.actor_network(inputs_norm_tensor, with_logprob = True)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean() + self.args.entropy_regularization*log_prob.mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # def sample_valid_goal(self,base_env):
    #     # return base_env.observation_space['observation'].sample()
    #     env = copy.deepcopy(base_env)
    #     env.reset()
    #     step_num = int(.9*env._max_episode_steps)
    #     step_num = 25
    #     for i in range(step_num):
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)

    #     return observation['observation']

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            # g = observation['desired_goal']
            # g = self.observation_space.sample()
            g = sample_valid_goal(self.env)
            # g = self.sample_random_goal()
            total_r = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    pi = self.actor_network.normed_forward(obs, g, deterministic=True)
                    actions = pi.detach().cpu().numpy().squeeze(axis=0)
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                # g = observation_new['desired_goal']
                total_r += self.sparse_reward_fn(obs, g)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(total_r)
            # total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        # local_success_rate = np.mean(total_success_rate[:, -1])
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
