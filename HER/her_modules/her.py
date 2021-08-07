import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.rng = np.random.default_rng()

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, rff=True):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        future_sampled_ag = episode_batch['ag'][episode_idxs, (t_samples + 1 + future_offset)]
        transitions['g'][her_indexes] = future_ag
        transitions['sampled_g'] = future_sampled_ag
        # to get the params to re-compute reward
        if rff:
              rff_visit = transitions['rff_visit'][her_indexes]
              self.rng.shuffle(rff_visit, axis=0)
              transitions['rff_visit'][her_indexes] = rff_visit
        #     rff_features = transitions['rff_visit'].shape[-1]
        #     transitions['rff_visit'][her_indexes] = np.random.uniform(-1/rff_features**.5, 1/rff_features**.5, 
        #                             size=transitions['rff_visit'].shape)[her_indexes]
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # import pdb
        # pdb.set_trace()

        return transitions


# class tdm_her_sampler:
#     def __init__(self, replay_strategy, replay_k, reward_func=None, vec_reward_func=None):
#         self.replay_strategy = replay_strategy
#         self.replay_k = replay_k
#         if self.replay_strategy == 'future':
#             self.future_p = 1 - (1. / (1 + replay_k))
#         else:
#             self.future_p = 0
#         self.reward_func = reward_func
#         self.vec_reward_func = vec_reward_func

#     def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
#         T = episode_batch['actions'].shape[1]
#         rollout_batch_size = episode_batch['actions'].shape[0]
#         batch_size = batch_size_in_transitions
#         # select which rollouts and which timesteps to be used
#         episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
#         t_samples = np.random.randint(T, size=batch_size)
#         transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
#         # her idx
#         her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
#         future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
#         future_offset = future_offset.astype(int)
#         future_t = (t_samples + 1 + future_offset)[her_indexes]
#         # replace go with achieved goal
#         future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
#         transitions['g'][her_indexes] = future_ag
#         # to get the params to re-compute reward
#         transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
#         transitions['vec_r'] = self.vec_reward_func(transitions['ag_next'], transitions['g'], None)
#         assert transitions['r'].shape[:-1] == transitions['vec_r'].shape[:-1]

#         transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

#         return transitions
