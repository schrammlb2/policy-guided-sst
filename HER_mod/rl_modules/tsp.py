import itertools
import os
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pdb 
from tspy import TSP
from tspy.solvers import TwoOpt_solver
from HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.hyperparams import NUM_GOALS


env = MultiGoalEnvironment("MultiGoalEnvironment") 
state = env.reset()
state_dim = state['observation'].shape[-1]
goal_dim = state['achieved_goal'].shape[-1]

num_goals = NUM_GOALS


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_path(n_goals):
    tsp = TSP()
    goals = [(np.random.rand(2)*2-1)*.95 for _ in range(n_goals)]
    dist = lambda a, b: np.sum((a-b)**2)**.5
    tsp.read_data(goals, dist)
    two_opt = TwoOpt_solver(initial_tour='NN', iter_num=1000)
    # with HiddenPrints():
    two_opt_tour = tsp.get_approx_solution(two_opt)
    path = [goals[ind] for ind in two_opt_tour]
    return path

def mean(lst):
    return sum(lst)/(len(lst)+.00001)

def std(lst):
    mn = mean(lst)
    return sum([(e-mn)**2 for e in lst])**.5/(len(lst)+.00001)

def confidence_interval(lst):
    return 2*std(lst)/(len(lst)+.00001)**.5


def l2(a,b):
    env = MultiGoalEnvironment("MultiGoalEnvironment") 
    return ((a-b)**2).sum()/env.step_size - 3


def q_metric(a,b, agent, vel_goal=False):
    #Pad with 0's if necessary
    if vel_goal: 
        goal_dim = state_dim
    else: 
        goal_dim = 2
    if a.shape[-1] < state_dim:
        a = np.concatenate([a, np.zeros(state_dim-a.shape[-1])])
    if b.shape[-1] < goal_dim:
        b = np.concatenate([b, np.zeros(goal_dim-b.shape[-1])])

    inp = np.concatenate([a, b], axis=-1) 
    inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
    # return -agent.critic_network(inp, agent.actor_network(inp))
    return -agent.critic.min_critic(inp, agent.actor_network(inp))


def evaluate_path(pos, path, metric):
    prev_loc = pos
    time = 0
    for goal in path:
        # if state_dim > goal_dim: 
        #     prev_loc = np.concatenate([pos, np.zeros(state_dim-goal_dim)])
        time += metric(prev_loc, goal)
        prev_loc = goal

    return time



def find_true_shortest_path(pos, goals):
    min_time = float("inf")
    min_path = None
    for path in itertools.permutations(goals):
        time = path_runner(pos, path)
        # print(time, path)
        if time < min_time: 
            min_time = time
            min_path = path

    return (min_time, min_path)



def list_dist(pos, goals, metric):
    pred_pairs = []
    for path in itertools.permutations(goals):
        predicted_time = evaluate_path(pos, path, metric)
        true_time = path_runner(pos, path)
        pred_pairs.append([predicted_time, true_time])
        print("predicted_time: " + str(predicted_time))
        print("true_time: " + str(true_time))
        print("-------------------")

    x = np.array(pred_pairs).transpose()
    import matplotlib.pyplot as plt
    plt.scatter(x[0], x[1])
    plt.show()
    import pdb
    pdb.set_trace()


    return (min_time, min_path)





def plot_path_costs(agent):
    run_time_list = []
    n=20
    for i in range(n):
    # return
        # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
        goals = generate_path(num_goals + 1)
        pos = goals[0]
        goals = goals[1:-1]
        # pos = np.random.rand(2)*2-1
        time_list = []
        min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=0, perm_search=False)
        # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=0)
        # print("Successful: " + str(successful))
        # print("Predicted time: " + str(min_time))
        # print("Empirical time: " + str(len(min_trajectory)))
        time_list.append(len(min_trajectory))
        # plot_path(pos, min_time, min_trajectory, min_path, successful, pass_vals)
        min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=5, perm_search=False)
        # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=5)
        # print("Successful: " + str(successful))
        # print("Predicted time: " + str(min_time))
        # print("Empirical time: " + str(len(min_trajectory)))
        time_list.append(len(min_trajectory))
        # plot_path(pos, min_time, min_trajectory, min_path, successful, pass_vals)
        min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=10, perm_search=False)
        # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=10)
        # print("Successful: " + str(successful))
        # print("Predicted time: " + str(min_time))
        # print("Empirical time: " + str(len(min_trajectory)))
        time_list.append(len(min_trajectory))
        # plot_path(pos, min_time, min_trajectory,  min_path, successful, pass_vals)
        min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=15, perm_search=False)
        # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=15)
        # print("Successful: " + str(successful))
        # print("Predicted time: " + str(min_time))
        # print("Empirical time: " + str(len(min_trajectory)))
        time_list.append(len(min_trajectory))
        # plot_path(pos, min_time, min_trajectory, min_path, successful, pass_vals)
        min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=20, perm_search=False)
        # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=20)
        # print("Successful: " + str(successful))
        # print("Predicted time: " + str(min_time))
        # print("Empirical time: " + str(len(min_trajectory)))
        time_list.append(len(min_trajectory))

        run_time_list.append(time_list)

    run_time_list = np.array(run_time_list).squeeze()
    mean_time = run_time_list.mean(axis=0)
    std_time = run_time_list.std(axis=0)
    ci = 2*std_time/(n**.5)
    steps = np.arange(mean_time.shape[-1])*5
    plt.plot(steps, mean_time)
    plt.fill_between(steps, mean_time+ci, mean_time-ci, alpha=.4)
    plt.xlabel("Gradient steps")
    plt.ylabel("Time to complete")
    plt.savefig(os.path.join('results', "Time to complete vs path optimization steps" + '.png'))
    plt.close()
    # plt.show()

    run_time_list = (run_time_list.transpose() - run_time_list[...,0]).transpose()
    mean_time = run_time_list.mean(axis=0)
    std_time = run_time_list.std(axis=0)
    ci = 2*std_time/(n**.5)
    steps = np.arange(mean_time.shape[-1])*5
    plt.plot(steps, mean_time)
    plt.fill_between(steps, mean_time+ci, mean_time-ci, alpha=.4)
    plt.xlabel("Gradient steps")
    plt.ylabel("Change vs 0 gradient steps")
    plt.savefig(os.path.join('results', "Change over no-optimization result vs path optimization steps" + '.png'))
    plt.close()



def plot_path_costs(agent):
    run_time_list = []
    n=10
    gd_step_list = [0,5,10,15,20, 30, 40]
    # gd_step_list = [0,5,10,20, 50, 80]
    successful_results = {}
    unsuccessful_results = {}
    for step_num in gd_step_list:
        successful_results[step_num] = []
        unsuccessful_results[step_num] = []

    times = []
    vel_len = []
    successful_list = []
    time_list = []

    for i in range(n):
        # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
        goals = generate_path(num_goals + 1)
        pos = goals[0]
        goals = goals[1:-1]

        for step_num in gd_step_list:
            min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=step_num, perm_search=False)
            total_vel = sum([(v.detach().numpy()**2).sum()**.5 for v in min_vel_path])/len(min_vel_path)
            vel_len.append(total_vel)
            time_list.append(len(min_trajectory))
            successful_list.append(1*successful)
            if successful: 
                successful_results[step_num].append(len(min_trajectory))
            else: 
                unsuccessful_results[step_num].append(len(min_trajectory))

    sr_means = np.array([mean(lst) for lst in successful_results.values()])
    sr_cis = np.array([confidence_interval(lst) for lst in successful_results.values()])

    usr_means = np.array([mean(lst) for lst in unsuccessful_results.values()])
    usr_cis = np.array([confidence_interval(lst) for lst in unsuccessful_results.values()])


    steps = np.array(gd_step_list)

    mean_time = sr_means
    ci = sr_cis
    plt.plot(steps, mean_time, label = "successful")
    plt.fill_between(steps, mean_time+ci, mean_time-ci, alpha=.4)
    plt.xlabel("Gradient steps")
    plt.ylabel("Time to complete")
    plt.legend()
    plt.savefig(os.path.join('results', "Cost of successful paths" + '.png'))
    plt.close()

    succ_rate = np.array([len(lst)/n for lst in successful_results.values()])
    plt.plot(steps, succ_rate)
    plt.xlabel("Gradient steps")
    plt.ylabel("Success rate")
    plt.savefig(os.path.join('results', "Success rate vs path optimization steps" + '.png'))
    plt.close()

    plt.scatter(vel_len, successful_list)
    plt.xlabel("Mean target Velocity")
    plt.ylabel("Successful")
    plt.savefig(os.path.join('results', "Success vs target velocity" + '.png'))
    plt.close()

    plt.scatter(vel_len, time_list)
    plt.xlabel("Mean target Velocity")
    plt.ylabel("Time to complete")
    plt.savefig(os.path.join('results', "Time to complete vs target velocity" + '.png'))
    plt.close()



def plot_path(agent, gd_step_list=[0], tru_min = False):
    env = MultiGoalEnvironment("asdf", time=True, vel_goal=False)
    # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]

    goals = generate_path(num_goals + 1)
    pos = goals[0]
    goals = goals[1:-1]
    for gd_steps in gd_step_list:
        if tru_min:
            min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_and_optimize_true_shortest_path(pos, goals, gd_steps=gd_steps, perm_search=False)
        else:
            min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = agent.find_shortest_path(pos, goals, gd_steps=gd_steps, perm_search=False)

        hit_points, env_goals = pass_vals
        g_x = [goal[0] for goal in min_path]
        g_y = [goal[1] for goal in min_path]
        traj_x = [t[0] for t in min_trajectory]
        traj_y = [t[1] for t in min_trajectory]
        min_dists = [min([((t-hp)**2).sum()**.5 for t in min_trajectory]) for hp in hit_points]
        # import pdb
        # pdb.set_trace()
        print("successful: " + str(successful))
        # print(min_time)
        print(len(min_trajectory))
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        plt.scatter(pos[0], pos[1], color='green', s=.1)
        # for pnt in min_path:
        for pnt in env_goals:
            circle = plt.Circle((pnt[0], pnt[1]), env.epsilon, facecolor='red', edgecolor='red')
            plt.gca().add_patch(circle)

        for pnt in hit_points:
            circle = plt.Circle((pnt[0], pnt[1]), .01, facecolor='green', edgecolor='green')
            plt.gca().add_patch(circle)

        for p, vp in zip(env_goals, [vp.detach().numpy() for vp in min_vel_path]):
            vp = vp/10
            plt.arrow(p[0], p[1], vp[0], vp[1])
        # plt.scatter(g_x, g_y, color='red', s=.1)
        plt.plot(traj_x, traj_y, color='blue')
        # plt.scatter([pos[0], pos[1]], color='green')
        # plt.scatter([g_x, g_y], color='red')
        # plt.plot([traj_x, traj_y], color='blue')
        plt.title("Path through environment")
        plt.savefig(os.path.join('results', "Tru_min: "+ str(tru_min) + "-- Path with %i planning steps.png" % gd_steps))
        plt.close()
    # plt.show()

def plot_cost_by_vel_goal(agent, last_static=True):
    # env = MultiGoalEnvironment("asdf", time=True, vel_goal=False)
    increments=11
    all_qs = []
    all_costs = []

    for i in range(5):
        goals = [torch.tensor(np.random.rand(2)*2-1, dtype=torch.float32) for i in range(num_goals)]
        pos = torch.tensor(np.random.rand(2)*2-1, dtype=torch.float32)
        time, vel_path = agent.evaluate_path(pos, goals, gd_steps=40, last_static=last_static)
        vel_target = vel_path[0]
        normed_vel_target = vel_target/(vel_target**2).sum()**.5
        q_list = []
        cost_list = []
        scale_list = []
        for i in range(increments):
            scale = (1-2/(increments-1)*i)
            scale_list.append(scale)
            curr_vel_target = scale*normed_vel_target
            curr_vel_path = [curr_vel_target] + vel_path[1:]
            q_cost = agent._eval_path_q_network(pos, goals, curr_vel_path).detach().numpy()
            q_list.append(q_cost)
            real_cost = agent.run_path(pos, goals, curr_vel_path)[0]
            cost_list.append(real_cost)

        all_qs.append(q_list)
        all_costs.append(cost_list)

    q_array = np.array(all_qs).squeeze()
    cost_array = np.array(all_costs).squeeze()
    scale_array = np.array(scale_list)

    q_mean, q_std = q_array.mean(axis=0), q_array.std(axis=0)
    cost_mean, cost_std = cost_array.mean(axis=0), cost_array.std(axis=0)

    plt.plot(scale_array, q_mean)
    plt.fill_between(scale_array, q_mean+q_std, q_mean-q_std, alpha=.4)
    plt.xlabel("Vector target scaling factor")
    plt.ylabel("Q Cost")
    plt.savefig(os.path.join('results', "Q cost, static final goal = " + str(last_static) + '.png'))
    plt.close()

    plt.plot(scale_array, cost_mean)
    plt.fill_between(scale_array, cost_mean + cost_std, cost_mean - cost_std, alpha=.4)
    plt.xlabel("Vector target scaling factor")
    plt.ylabel("Actual Cost")
    plt.savefig(os.path.join('results', "Actual cost, static final goal = " + str(last_static)+ '.png'))
    plt.close()


def find_vector_field(agent, next_goals= [], last_static=False):
    increments=11
    pos = torch.tensor(np.zeros(2), dtype=torch.float32)
    loc_list = []
    vec_list = []
    for i in range(increments):
        for j in range(increments):
            x_goal = (1-2/(increments-1)*i)
            y_goal = (1-2/(increments-1)*j)
            first_goal = np.array([x_goal, y_goal])
            loc_list.append(first_goal)
            goals = [torch.tensor(first_goal)] + next_goals

            time, vel_path = agent.evaluate_path(pos, goals, gd_steps=40, last_static=last_static)
            vel_target = vel_path[0]

            vec_list.append(vel_target.detach().numpy())

    return loc_list, vec_list


def plot_gradient_vector_field(agent):
    # env = MultiGoalEnvironment("asdf", time=True, vel_goal=False)
    scale = 20

    loc_list, vec_list = find_vector_field(agent)
    for loc, vec in zip(loc_list, vec_list):
        vec = vec/scale
        plt.arrow(loc[0], loc[1], vec[0], vec[1])
    plt.savefig(os.path.join('results', "Vector Field -- Single Goal" + '.png'))
    plt.close()

    loc_list, vec_list = find_vector_field(agent, next_goals=[torch.ones(2)])
    for loc, vec in zip(loc_list, vec_list):
        vec = vec/scale
        plt.arrow(loc[0], loc[1], vec[0], vec[1])
    plt.savefig(os.path.join('results', "Vector Field -- Two Goals" + '.png'))
    plt.close()


    loc_list, vec_list = find_vector_field(agent, next_goals=[torch.ones(2)], last_static=True)
    for loc, vec in zip(loc_list, vec_list):
        vec = vec/scale
        plt.arrow(loc[0], loc[1], vec[0], vec[1])
    plt.savefig(os.path.join('results', "Vector Field -- Two Goals, Optimize only first goal" + '.png'))
    plt.close()


# def new_plot_path_costs(args, agent, vel_goal=False):
#     import matplotlib.pyplot as plt
#     solution_cost_list = []

#     learned_metric = lambda a,b: q_metric(a, b, agent, vel_goal=vel_goal)
#     metric = learned_metric
#     # metric_list = [l2, single_goal_run, learned_metric]
#     metric_list = [single_goal_run, learned_metric]
#     agent_list = [SimpleAgent(), NNAgent(agent)]
#     # evaluator_list = path_runner
#     cost_pairs = []
#     true_cost_list = []
#     optimal_paths = []

#     for i in range(50):
#         goals = [np.random.rand(2)*2-1 for i in range(5)]
#         pos = np.random.rand(2)*2-1

#         solutions = [find_shortest_path(pos, goals, metric) for metric in metric_list]
#         predicted_costs = [sol[0] for sol in solutions]
#         predicted_paths = [sol[1] for sol in solutions]
#         # true_costs = [path_runner(pos, path) for path in predicted_paths]
#         true_costs = [path_runner(pos, path, agent=agent) for path, agent in zip(predicted_paths, agent_list)]
#         # true_costs = [evaluate_path(pos, path, single_goal_run) for path in predicted_paths]
#         new_cost_pairs = [(pred, tru) for pred, tru in zip(predicted_costs, true_costs)]
#         true_cost_list.append(true_costs)
#         cost_pairs.append(new_cost_pairs)
#         optimal_paths.append(find_true_shortest_path(pos, goals)[0])
#         # import pdb
#         # pdb.set_trace()
#         # optimal_paths.append(pre)
#         # optimal_paths.append(predicted_costs[1])

#     cost_pairs = np.array(cost_pairs)
#     optimal_paths = np.array(optimal_paths)
#     # analytic_cost_plot = plt.
#     pred_cost = cost_pairs[...,0]
#     true_cost = cost_pairs[...,1]
#     for i, title in zip(range(len(metric_list)), ["L2 Norm", "Single Goal Cost", "Q Function"]):
#         plt.scatter(pred_cost[:,i], true_cost[:,i])
#         plt.xlabel("Predicted Costs")
#         plt.ylabel("True Costs")
#         plt.title(title)
#         plt.savefig(os.path.join('results', "Predicted vs true_costs: " + title + '.png'))
#         plt.close()
#         # plt.show()

#     min_path = optimal_paths.min()
#     max_path = optimal_paths.max()
#     line = np.array([min_path, max_path])
#     # import pdb
#     # pdb.set_trace()

#     for i, title in zip(range(len(metric_list)), ["L2 Norm", "Single Goal Cost", "Q Function"]):
#         # plt.scatter(optimal_paths, true_cost[:,i])
#         plt.plot(line,line)
#         plt.scatter(true_cost[:,1], true_cost[:,i])
#         plt.xlabel("Optimal solution")
#         plt.ylabel("Cost of solution for " + title)
#         plt.title(title)
#         # plt.savefig(os.path.join('results', title + '.png'))
#         plt.savefig(os.path.join('results', "Cost of predicted path : " + title + '.png'))
#         plt.close()
#         # plt.show()



# def plot_path_costs(args, agent, vel_goal=False):
#     import matplotlib.pyplot as plt
#     solution_cost_list = []

#     learned_metric = lambda a,b: q_metric(a, b, agent, vel_goal=vel_goal)
#     metric_list = [l2, single_goal_run, learned_metric]
#     cost_pairs = []
#     true_cost_list = []
#     optimal_paths = []

#     for i in range(50):
#         goals = [np.random.rand(2)*2-1 for i in range(5)]
#         pos = np.random.rand(2)*2-1

#         solutions = [find_shortest_path(pos, goals, metric) for metric in metric_list]
#         predicted_costs = [sol[0] for sol in solutions]
#         predicted_paths = [sol[1] for sol in solutions]
#         # true_costs = [path_runner(pos, path) for path in predicted_paths]
#         true_costs = [evaluate_path(pos, path, single_goal_run) for path in predicted_paths]
#         new_cost_pairs = [(pred, tru) for pred, tru in zip(predicted_costs, true_costs)]
#         true_cost_list.append(true_costs)
#         cost_pairs.append(new_cost_pairs)
#         optimal_paths.append(find_true_shortest_path(pos, goals)[0])
#         # import pdb
#         # pdb.set_trace()
#         # optimal_paths.append(pre)
#         # optimal_paths.append(predicted_costs[1])

#     cost_pairs = np.array(cost_pairs)
#     optimal_paths = np.array(optimal_paths)
#     # analytic_cost_plot = plt.
#     pred_cost = cost_pairs[...,0]
#     true_cost = cost_pairs[...,1]
#     for i, title in zip(range(len(metric_list)), ["L2 Norm", "Single Goal Cost", "Q Function"]):
#         plt.scatter(pred_cost[:,i], true_cost[:,i])
#         plt.xlabel("Predicted Costs")
#         plt.ylabel("True Costs")
#         plt.title(title)
#         plt.savefig(os.path.join('results', "Predicted vs true_costs: " + title + '.png'))
#         plt.close()
#         # plt.show()

#     min_path = optimal_paths.min()
#     max_path = optimal_paths.max()
#     line = np.array([min_path, max_path])
#     # import pdb
#     # pdb.set_trace()

#     for i, title in zip(range(len(metric_list)), ["L2 Norm", "Single Goal Cost", "Q Function"]):
#         # plt.scatter(optimal_paths, true_cost[:,i])
#         plt.plot(line,line)
#         plt.scatter(true_cost[:,1], true_cost[:,i])
#         plt.xlabel("Optimal solution")
#         plt.ylabel("Cost of solution for " + title)
#         plt.title(title)
#         # plt.savefig(os.path.join('results', title + '.png'))
#         plt.savefig(os.path.join('results', "Cost of predicted path : " + title + '.png'))
#         plt.close()
#         # plt.show()


