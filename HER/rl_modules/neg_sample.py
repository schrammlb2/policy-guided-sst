import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from HER.mpi_utils.mpi_utils import sync_networks, sync_grads
from HER.mpi_utils.normalizer import normalizer
# from HER.mpi_utils.normalizer import variant_normalizer as normalizer
from HER.rl_modules.replay_buffer import replay_buffer
from HER.rl_modules.sac_models import actor, critic, dual_critic
from HER.her_modules.her import her_sampler

"""
ddpg with HER (MPI-version)

"""
# critic_constructor = critic
critic_constructor = dual_critic


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
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

        # self.dense_reward_fn = lambda o, g, x=None: -np.mean((o-g)**2, axis=-1)**.5*50
        # power = 2
        # self.dense_reward_fn = lambda o, g, x=None: -np.mean((o-g)**power, axis=-1)*50**power
        self.dense_reward_fn = lambda o, g, x=None: -np.mean((o-g)**2, axis=-1)*50**2

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.dense_reward_fn)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
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

        # self.negative_sampling = True
        self.negative_sampling = False
        self.neg_lambda = .0001
        self.neg_std = 5.0
        self.neg_ave = 0
        self.n=1
        self.mean = 0
        self.std = 0

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
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
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
            success_rate, ave_reward = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:

                # print('[{}] epoch is: {}, eval success rate is: {:.3f}, average reward is {:.3f}'.format(datetime.now(), epoch, success_rate, ave_reward))
                # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                #             self.model_path + '/model.pt')
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, average reward is {:.3f}, average neg value is {:.2f}'.format(
                    datetime.now(), epoch, success_rate, ave_reward, self.neg_ave))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

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
            log_prob_next = log_prob_next.unsqueeze(dim=-1)
            assert log_prob_next.shape == r_tensor.shape
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next) + self.args.entropy_regularization*log_prob_next 
            q_next_value = q_next_value.detach()
            assert q_next_value.shape == r_tensor.shape

            nonterminal = -self.env.compute_reward(transitions['g'],transitions['ag'], None)
            nonterminal_tensor = torch.tensor(nonterminal, dtype=torch.float32).unsqueeze(dim=-1)
            assert q_next_value.shape == nonterminal_tensor.shape

            target_q_value = r_tensor + self.args.gamma * q_next_value*nonterminal_tensor #* (-r_tensor) 
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            
            self.std += inputs_norm_tensor.std(dim=0)
            self.mean += inputs_norm_tensor.mean(dim=0)
            self.n += 1
            if self.negative_sampling:
                negative_samples = torch.normal(0.,1., inputs_norm_tensor.shape)*self.neg_std*(self.std/self.n)
                negative_actions = self.actor_target_network(negative_samples, with_logprob = False)
        # the q loss
        # real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        # critic_loss = (target_q_value - real_q_value).pow(2).mean()
        real_q_1, real_q_2 = self.critic_network.dual(inputs_norm_tensor, actions_tensor)
        q_loss = (target_q_value - real_q_1).pow(2).mean() + (target_q_value - real_q_2).pow(2).mean()

        
        if self.negative_sampling:
            neg_q1, neg_q2 = self.critic_network.dual(negative_samples, negative_actions)
            cut_val = -50
            # reg_loss = self.neg_lambda*((cut_val - neg_q1)**2 + (cut_val - neg_q2)**2).mean()
            reg_loss = self.neg_lambda*(torch.abs(cut_val - neg_q1) + torch.abs(cut_val - neg_q2)).mean()

            critic_loss = q_loss + reg_loss

            self.neg_ave = neg_q1.mean().clone().detach().numpy()
        else: 
            critic_loss = q_loss
            self.neg_ave = 0
        # if np.random.rand() < .001: 
        #     import pdb
        #     pdb.set_trace()

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

    # do the evaluation
    # def _eval_agent(self):
    #     total_success_rate = []
    #     for _ in range(self.args.n_test_rollouts):
    #         per_success_rate = []
    #         observation = self.env.reset()
    #         obs = observation['observation']
    #         g = observation['desired_goal']
    #         total_r = 0
    #         for _ in range(self.env_params['max_timesteps']):
    #             with torch.no_grad():
    #                 pi = self.actor_network.normed_forward(obs, g, deterministic=True)
    #                 # input_tensor = self._preproc_inputs(obs, g)
    #                 # pi = self.actor_network(input_tensor, deterministic=True)
    #                 # convert the actions
    #                 # import pdb
    #                 # pdb.set_trace()
    #                 # actions = pi.detach().cpu().numpy().squeeze()
    #                 actions = pi.detach().cpu().numpy().squeeze(axis=0)
    #             observation_new, r, _, info = self.env.step(actions)
    #             total_r += r
    #             obs = observation_new['observation']
    #             g = observation_new['desired_goal']
    #             per_success_rate.append(info['is_success'])
    #         # total_success_rate.append(per_success_rate)
    #         total_success_rate.append(total_r)
    #     total_success_rate = np.array(total_success_rate)
    #     # local_success_rate = np.mean(total_success_rate[:, -1])
    #     local_success_rate = np.mean(total_success_rate)
    #     global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    #     return global_success_rate / MPI.COMM_WORLD.Get_size()


    def _eval_agent(self, verbose=False):
        total_success_rate = []
        total_reward = []
        run_num = self.args.n_test_rollouts
        if verbose: 
            run_num = 1
        for _ in range(run_num):
            total_r = 0
            # per_success_rate = []
            success = 0
            observation = self.env.reset()

            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
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
        # import pdb
        # pdb.set_trace()
        total_success_rate = np.array(total_success_rate)
        total_reward = np.array(total_reward)
        # local_success_rate = np.mean(total_success_rate[:, -1])
        local_success_rate = np.mean(total_success_rate)
        local_reward = np.mean(total_reward)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_reward / MPI.COMM_WORLD.Get_size()


    def print_value_sequence(self, value_estimator, forced_state=None):
        print("---------------------------------------------------------")
        print("Agent sequence")
        print("---------------------------------------------------------")
        observation = self.env.reset()
        if type(forced_state) != type(None):
            s = [0] + forced_state.tolist()
            # self.sim.set_state_from_flattened(np.array(state))
            self.env.sim.set_state_from_flattened(s)
            self.env.sim.forward()
            observation = self.env._get_obs()

        obs = observation['observation']
        g = observation['desired_goal']
        ag = observation['achieved_goal']
        value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        total_r = 0
        print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Total Reward: %2.1f" % (value, total_r))
        for _ in range(10):#self.env_params['max_timesteps']):
            with torch.no_grad():
                input_tensor = self._preproc_inputs(obs, g)
                pi = self.actor_network(input_tensor)
                # convert the actions
                actions = pi.detach().cpu().numpy().squeeze(axis=0)
            observation, r, done, info = self.env.step(actions)
            total_r += r

            obs = observation['observation']
            g = observation['desired_goal']
            ag = observation['achieved_goal']
            # per_success_rate.append(info['is_success'])
            success = info['is_success']
            value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
            print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Total Reward: %2.1f" % (value, total_r))
            if done: 
                break




    def print_gd_value_sequence(self, value_estimator, forced_state=None, verbose = False):
        if verbose:
            print("---------------------------------------------------------")
            print("Gradient descent sequence")
            print("---------------------------------------------------------")
        observation = self.env.reset()
        def set_state(env, state):
            s = [0] + state.tolist()
            env.env.sim.set_state_from_flattened(np.array(s))
            # self.env.sim.set_state_from_flattened(s)
            env.env.sim.forward()
            observation = env.env._get_obs()
            return observation

        if type(forced_state) != type(None):
            observation = set_state(env, forced_state)

        obs = observation['observation']
        g = observation['desired_goal']
        ag = observation['achieved_goal']

        hill_climbing=True
        hill_climbing=False
        
        value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        s = torch.tensor(obs, requires_grad=True, dtype=torch.float32)
        total_r = 0
        n=20
        mean = self.mean[:-3]/self.n
        std = self.std[:-3]/self.n
        normed_s, normed_g = value_estimator.norm(s, torch.tensor(g, dtype=torch.float32))
        normed_s = normed_s.detach().requires_grad_()
        def mean_val(state, n=20): 
            return sum([value_estimator(state + torch.normal(0.,.01, state.shape), 
                normed_g, norm=False) for _ in range(n)])/n

        initial_ag = ag
        initial_goal = g

        steps = 50
        if hill_climbing:
            # value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
            # s = torch.tensor(obs, dtype=torch.float32)
            # total_r = 0
            n=200
            # mean_val = lambda s: sum([value_estimator(s + torch.normal(0.,.03, s.shape), 
            #     torch.tensor(g, dtype=torch.float32)) for _ in range(n)])/n
            if verbose: 
                print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Total Reward: %2.1f" % (value, total_r))
            for i in range(steps):
                # rand_vec = [torch.normal(0.,.05, s.shape)*(self.std/self.n)[:-3] for _ in range(n)] + [torch.zeros(s.shape)]
                rand_vec = [torch.normal(0.,.03, s.shape) for _ in range(n)] + [torch.zeros(s.shape)]
                # vals = [-value_estimator(s + rand_vec[i], torch.tensor(g, dtype=torch.float32)).detach() for i in range(n+1)]
                # vals = [-value_estimator(s + vec, torch.tensor(g, dtype=torch.float32)).detach() for vec in rand_vec]
                vals = [-value_estimator(normed_s + vec, normed_g, norm=False).detach() for vec in rand_vec]
                min_i= np.argmin(vals)
                # s = s + rand_vec[min_i]
                normed_s= normed_s + rand_vec[min_i]
                # value = mean_val(s)

                # observation_new = set_state(self.env, s.detach().numpy())
                # observation_new = set_state(self.env, normed_s.detach().numpy())
                new_s, new_g = value_estimator.denorm(normed_s, normed_g)
                observation_new = set_state(self.env, new_s.detach().numpy())

                obs = observation_new['observation']
                g = observation_new['desired_goal']
                ag = observation_new['achieved_goal']

                total_r += self.env.compute_reward(g, ag, None)
                current_reward = self.dense_reward_fn(g, ag)

                # value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
                # value = value_estimator(s, torch.tensor(g, dtype=torch.float32))
                value = value_estimator(normed_s, normed_g, norm=False).detach()
                if verbose: 
                    print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Current reward: %4.2f \t Total Reward: %2.1f" % (value, current_reward, total_r))

        else:
            opt = torch.optim.SGD([normed_s], lr=.01)
            # opt = torch.optim.SGD([normed_s], lr=.000001)
            opt = torch.optim.Adam([normed_s], lr=.05)
                # return sum([value_estimator(s + torch.normal(0.,.01, s.shape)*(std), 
                # torch.tensor(g, dtype=torch.float32)) for _ in range(n)])/n
            if verbose:
                print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Total Reward: %2.1f" % (value, total_r))
            for i in range(steps):
                opt.zero_grad()
                # value = value_estimator(s, torch.tensor(g, dtype=torch.float32))
                # value = value_estimator(normed_s, normed_g, norm=False)
                value = mean_val(normed_s)
                # reg_loss = (((s - mean)/std)**2).sum()
                # normed_s, normed_g = value_estimator.norm(s, torch.tensor(g, dtype=torch.float32))
                reg_loss = ((normed_s)**2).sum()
                loss = -value + 1*reg_loss
                loss.backward()
                # normed_s.grad*=std
                opt.step()

                # observation_new = set_state(self.env, s.detach().numpy())
                new_s, new_g = value_estimator.denorm(normed_s, normed_g)
                observation_new = set_state(self.env, new_s.detach().numpy())

                obs = observation_new['observation']
                g = observation_new['desired_goal']
                ag = observation_new['achieved_goal']
                # per_success_rate.append(info['is_success'])
                # success = info['is_success']

                total_r += self.env.compute_reward(g, ag, None)
                current_reward = self.dense_reward_fn(g, ag)

                value = value_estimator(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
                # print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Total Reward: %2.1f" % (value, total_r))
                if verbose: 
                    print("AG: "+ str(ag) + "\tGoal: "+ str(g) + "\t Value: %2.2f \t Current reward: %2.2f \t Total Reward: %2.1f" % (value, current_reward, total_r))

        print('-------------------------------------------------------------------')
        print("Initial AG: " + str(initial_ag) + 
            "\nDesired Goal: " + str(initial_goal) +
            "\nFinal AG: " + str(ag) + 
            "\tTotal reward: %2.1f" % (total_r))
        return total_r