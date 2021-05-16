from HER_mod.rl_modules.tsp import *
from HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.value_map import *

class Hook:
    def __init__(self, args=None):
        raise NotImplementedError

    def run(self, agent):
        raise NotImplementedError

    def finish(self):
        pass


class ValueMapHook(Hook):
    def __init__(self, args=None, target_vels = [[0,0]]):
        self.epoch = 0
        self.target_vels = target_vels

    def run(self, agent):
        # q_value_map(agent.critic.min_critic, agent.actor_network, title= "HER DDPG value map, epoch " + str(self.epoch))
        for target_vel in self.target_vels:
            q_value_map(agent.planning_critic.min_critic, agent.actor_network, title= "HER DDPG value map, epoch " + str(self.epoch) + ", v = ", target_velocity = target_vel)
        self.epoch += 1

    def finish(self):
        pass


class VelocityValueMapHook(Hook):
    def __init__(self, args=None, target_vels = [[0,0]]):
        self.epoch = 0
        self.target_vels = target_vels

    def run(self, agent):
        # q_value_map(agent.critic.min_critic, agent.actor_network, title= "HER DDPG value map, epoch " + str(self.epoch))
        for target_vel in self.target_vels:
            q_velocity_value_map(agent.planning_critic.min_critic_target, agent.actor_network, title= "Velocity value map, epoch " + str(self.epoch))
        self.epoch += 1

    def finish(self):
        pass

class EmpiricalVelocityValueMapHook(Hook):
    def __init__(self, args=None, target_vels = [[0,0]]):
        self.epoch = 0
        self.target_vels = target_vels

    def run(self, agent):
        # q_value_map(agent.critic.min_critic, agent.actor_network, title= "HER DDPG value map, epoch " + str(self.epoch))
        # for target_vel in self.target_vels:
        empirical_velocity_value_map(agent, title= "Empirical velocity value map, epoch " + str(self.epoch))
        # q_velocity_value_map(agent.planning_critic.min_critic_target, agent.actor_network, title= "Empirical velocity value map, epoch " + str(self.epoch))
        self.epoch += 1

    def finish(self):
        pass


class DiffMapHook(Hook):
    def __init__(self, args=None):
        self.epoch = 0

    def run(self, agent):
        # q_value_map(agent.critic.min_critic, agent.actor_network, title= "HER DDPG value map, epoch " + str(self.epoch))
        q_diff_map(agent.planning_critic.min_critic, agent.actor_network, title= "Diff map, epoch " + str(self.epoch))
        self.epoch += 1

    def finish(self):
        pass


class DistancePlottingHook(Hook):
    def __init__(self, args=None):
        # self.super()

        self.ratio_list = []

        iters = 50
        self.pos_list = [np.random.rand(2)*2-1 for _ in range(iters)]
        self.goal_list = [[np.random.rand(2)*2-1 for _ in range(5)] for _ in range(iters)]

    def run(self, agent):
        metric = lambda a,b: q_metric(a, b, agent)

        mean = 0
        for pos, goals in zip(self.pos_list, self.goal_list):
            cost, path = find_shortest_path(pos, goals, metric)
            true_cost = evaluate_path(pos, path, single_goal_run)
            mean += cost/true_cost

        mean /= len(self.pos_list)
        self.ratio_list.append(mean)

    def finish(self):
        import matplotlib.pyplot as plt
        plt.plot(list(range(len(self.ratio_list))), self.ratio_list)
        plt.ylabel("Ratio of predicted cost to true cost")
        plt.xlabel("Epoch")
        plt.savefig(os.path.join('results', "Ratio of predicted cost to true cost.png"))
        # plt.show()
        plt.close()

# class PlotPathCostsHook(Hook):
#     def __init__(self, args=None, vel_goal=False):
#         self.args = args
#         self.vel_goal = vel_goal

#     def run(self, agent):
#         self.agent = agent

#     def finish(self):
#         plot_path_costs(self.args, self.agent, vel_goal=self.vel_goal)

class PlotPathCostsHook(Hook):
    def __init__(self, args=None):
        self.args = args
        # self.vel_goal = vel_goal

    def run(self, agent):
        self.agent = agent

    def finish(self):
        plot_path_costs(self.agent)
        # return plot_path_costs(self.agent)

class GradientDescentShortestPathHook(Hook):
    def __init__(self, args = ([0], False)):
        gd_step_list = args[0]
        tru_min = args[1]
        self.gd_step_list = gd_step_list
        self.tru_min = tru_min
        # self.vel_goal = vel_goal

    def run(self, agent):
        self.agent = agent

    def finish(self):
        plot_path(self.agent, gd_step_list=self.gd_step_list, tru_min=self.tru_min)
