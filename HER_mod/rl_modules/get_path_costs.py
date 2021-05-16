import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats
from HER_mod.rl_modules.tsp import generate_path
from HER_mod.rl_modules.hyperparams import NUM_GOALS, NUM_AGENTS

gd_step_list = [0,2,5, 10, 20, 40]
# NUM_AGENTS = 3
N=200

def get_path_costs(train_pos_agent, train_vel_agent, perm_search=True):
    pos_run_time_list = []
    vel_run_time_list = []
    # gd_step_list = [0,5,10]
    num_agents = NUM_AGENTS
    num_goals=NUM_GOALS
    n=N
    # gd_step_list = [0,1]
    # num_agents = 2
    # num_goals=2
    # n=2
    pos_time_list = []
    vel_time_list = []
    for _ in range(num_agents):
        pos_agent = train_pos_agent()
        vel_agent = train_vel_agent()
        pos_agent_time_list = []
        vel_agent_time_list = []
        for i in range(n):
            # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
            # pos = np.random.rand(2)*2-1

            goals = generate_path(num_goals + 1)
            pos = goals[0]
            goals = goals[1:-1]
            # pos_agent_time_list = []
            min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = pos_agent.find_shortest_path(pos, goals, gd_steps=0, perm_search=perm_search)
            pos_test_time_list = [len(min_trajectory)]*len(gd_step_list)
            pos_agent_time_list.append(pos_test_time_list)


            vel_test_time_list = []
            for gd_steps in gd_step_list:
                min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.find_shortest_path(pos, goals, gd_steps=gd_steps, perm_search=perm_search)
                vel_test_time_list.append(len(min_trajectory))
            vel_agent_time_list.append(vel_test_time_list)
        
        pos_time_list.append(pos_agent_time_list)
        vel_time_list.append(vel_agent_time_list)
    
    vel_time_list = np.array(vel_time_list).squeeze()
    pos_time_list = np.array(pos_time_list).squeeze()

    relative_time_change = (vel_time_list-pos_time_list)/pos_time_list
    relative_time_change = np.mean(relative_time_change, axis=1)

    try:
        pickle.dump(vel_time_list, open("velocity_target.pkl", 'wb'))
        pickle.dump(pos_time_list, open("no_velocity_target.pkl", 'wb'))
        pickle.dump(relative_time_change, open("relative_time_change.pkl", 'wb'))
    except:
        print("pickle failure")
        import pdb
        pdb.set_trace()

    mean = relative_time_change.mean(axis=0)
    t_score = stats.t.ppf(.975, num_agents)
    ci = t_score*relative_time_change.std(axis=0)/(num_agents**.5)
    steps = np.array(gd_step_list)

    plt.plot(steps, mean)
    plt.fill_between(steps, mean+ci, mean-ci, alpha=.4)
    plt.xlabel("Gradient steps")
    plt.ylabel("Relative Improvement vs standard HER")
    plt.title("Relative Improvement")
    plt.savefig(os.path.join('results', "Relative Improvement" + '.png'))
    plt.close()
    # import pdb
    # pdb.set_trace()



# def method_comparison(train_pos_agent, train_vel_agent):
#     # method_list = ['random search', "gradient descent", "gradient descent (40 steps)", "random", "0 velocity target"]
#     method_list = ['random search', "gradient descent",  "random", "0 velocity target"]

#     method_runtime_dict = {'greedy': []}
#     for method in method_list:
#         method_runtime_dict[method] = [] 


#     num_agents = NUM_AGENTS
#     num_goals=NUM_GOALS
#     n=N

#     pos_time_list = []
#     vel_time_list = []
#     for _ in range(num_agents):
#         pos_agent = train_pos_agent()
#         vel_agent = train_vel_agent()

#         for method in method_runtime_dict.keys():
#             method_runtime_dict[method].append([])

#         for i in range(n):
#             # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
#             # pos = np.random.rand(2)*2-1
#             goals = generate_path(num_goals + 1)
#             pos = goals[0]
#             goals = goals[1:-1]
#             # pos_agent_time_list = []
#             min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = pos_agent.select_path(pos, goals, method="0 velocity target")
#             # pos_test_time_list = [len(min_trajectory)]*len(gd_step_list)
#             method_runtime_dict['greedy'][-1].append(len(min_trajectory))


#             # vel_test_time_list = []
#             for method in method_list:
#                 min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.select_path(pos, goals, method=method)
#                 method_runtime_dict[method][-1].append(len(min_trajectory))
#             # vel_agent_time_list.append(vel_test_time_list)


#     greedy = method_runtime_dict['greedy']
#     method_runtime_dict = {method: np.array(method_runtime_dict[method]) for method in method_runtime_dict.keys()}
#     performance_dict = {method: (method_runtime_dict[method].mean(), 2*(method_runtime_dict[method].mean(axis=-1)).std()/(num_agents**.5)) for method in method_runtime_dict.keys()}
#     relative_time_dict = {method: (method_runtime_dict[method] - greedy)/greedy for method in method_list}
#     improvement_dict = {method: (relative_time_dict[method].mean(), 2*(relative_time_dict[method].mean(axis=-1)).std()/(num_agents**.5)) for method in method_list}


#     performance_list = [performance_dict[m][0] for m in method_runtime_dict.keys()]
#     performance_ci_list = [performance_dict[m][1] for m in method_runtime_dict.keys()]
#     relative_time_list = [improvement_dict[m][0] for m in method_list]
#     relative_time_ci_list = [improvement_dict[m][1] for m in method_list]




#     plt.xticks(range(len(method_runtime_dict.keys())), list(method_runtime_dict.keys()))
#     plt.xlabel("Method")
#     plt.ylabel('Time to complete')
#     plt.title('Comparison of velocity target-setting methods')
#     plt.bar(range(len(performance_list)), performance_list, yerr=performance_ci_list) 
#     plt.savefig(os.path.join('results', "Method comparison -- Performance" + '.png'))
#     plt.close()


#     plt.xticks(range(len(method_list)), method_list)
#     plt.xlabel("Method")
#     plt.ylabel('Cost reduction over greedy baseline')
#     plt.title('Comparison of velocity target-setting methods')
#     plt.bar(range(len(relative_time_list)), relative_time_list, yerr=relative_time_ci_list) 
#     plt.savefig(os.path.join('results', "Method comparison -- Relative Improvement" + '.png'))
#     plt.close()



def method_comparison(train_pos_agent, train_vel_agent):
    method_list = ['random search', "gradient descent", "gradient descent (40 steps)", "random", "0 velocity target"]
    # method_list = ['random search', "gradient descent",  "random", "0 velocity target"]

    method_runtime_dict = {'greedy': []}
    for method in method_list:
        method_runtime_dict[method] = [] 


    num_agents = NUM_AGENTS
    num_goals=NUM_GOALS
    n=N

    pos_time_list = []
    vel_time_list = []

    failed_counter_dict = {'greedy': 0}
    for method in method_list:
        failed_counter_dict[method] = 0


    for _ in range(num_agents):
        pos_agent = train_pos_agent()
        vel_agent = train_vel_agent()

        for method in method_runtime_dict.keys():
            method_runtime_dict[method].append([])

        for i in range(n):
            # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
            # pos = np.random.rand(2)*2-1
            goals = generate_path(num_goals + 1)
            pos = goals[0]
            goals = goals[1:-1]
            # pos_agent_time_list = []
            min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = pos_agent.select_path(pos, goals, method="0 velocity target")
            # pos_test_time_list = [len(min_trajectory)]*len(gd_step_list)
            if successful: 
                method_runtime_dict['greedy'][-1].append(len(min_trajectory))
            else: 
                method_runtime_dict['greedy'][-1].append("NULL")
                failed_counter_dict['greedy'] += 1


            # vel_test_time_list = []
            for method in method_list:
                min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.select_path(pos, goals, method=method)
                if successful: 
                    method_runtime_dict[method][-1].append(len(min_trajectory))
                else: 
                    method_runtime_dict[method][-1].append("NULL")
                    failed_counter_dict[method] += 1
            # vel_agent_time_list.append(vel_test_time_list)

    success_rates = {method: 1-failed_counter_dict[method]/(num_agents*n) for method in failed_counter_dict.keys()}

    greedy = method_runtime_dict['greedy']
    agent_performance_dict = {}
    mean_performance_dict = {}
    ci_performance_dict = {}

    improvement_dict = {}
    mean_improvement_dict = {}
    ci_improvement_dict = {}
    t_score = stats.t.ppf(.975, num_agents)


    for method in method_runtime_dict.keys(): 
        agent_performance_dict[method] = [[time for time in agent_list  if time != "NULL"]  for agent_list in method_runtime_dict[method]]
        agent_performance_dict[method] = [sum(agent_list)/len(agent_list)  for agent_list in agent_performance_dict[method]]
        mean = sum(agent_performance_dict[method])/len(agent_performance_dict[method])
        mean_performance_dict[method] = mean
        ci_performance_dict[method]   = t_score*sum([(v-mean)**2 for v in agent_performance_dict[method]])**.5/len(agent_performance_dict[method])

        improvement_list = []
        mean_list = []
        for agent_ind in range(num_agents):
            agent_list = method_runtime_dict[method][agent_ind]
            greedy_list = greedy[agent_ind]
            improvement_list.append([(agent_list[i] - greedy_list[i])/greedy_list[i] for i in range(n) if (agent_list[i] != "NULL" and greedy_list[i]!= "NULL")])
            mean_list.append(sum(improvement_list[agent_ind])/len(improvement_list[agent_ind]))

        mean = sum(mean_list)/len(mean_list)
        mean_improvement_dict[method] = mean
        ci_improvement_dict[method]   = t_score*sum([(v-mean)**2 for v in mean_list])**.5/len(mean_list)

        # agent_improvement_dict[method] = [[(time - greedy_time)/greedy_time for time in agent_list  if time != "NULL"]  for agent_list in method_runtime_dict[method]]
        # agent_performance_dict[method] = [sum(agent_list)/len(agent_list)  for agent_list in agent_performance_dict[method]]
        # mean_performance_dict[method] = sum(agent_performance_dict[method])/len(agent_performance_dict[method])
        # ci_performance_dict[method]   = 2*sum([(v-mean)**2 for v in agent_performance_dict[method]])**.5/len(agent_performance_dict[method])
    # method_runtime_dict = {method: np.array(method_runtime_dict[method]) for method in method_runtime_dict.keys()}

    # mean_performance_dict = {method: method_runtime_dict[method] for method in method_runtime_dict.keys()}
    # relative_time_dict = {method: (method_runtime_dict[method] - greedy)/greedy for method in method_list}
    # improvement_dict = {method: (relative_time_dict[method].mean(), 2*(relative_time_dict[method].mean(axis=-1)).std()/(num_agents**.5)) for method in method_list}


    # greedy = method_runtime_dict['greedy']
    # method_runtime_dict = {method: np.array(method_runtime_dict[method]) for method in method_runtime_dict.keys()}
    # performance_dict = {method: (method_runtime_dict[method].mean(), 2*(method_runtime_dict[method].mean(axis=-1)).std()/(num_agents**.5)) for method in method_runtime_dict.keys()}
    # relative_time_dict = {method: (method_runtime_dict[method] - greedy)/greedy for method in method_list}
    # improvement_dict = {method: (relative_time_dict[method].mean(), 2*(relative_time_dict[method].mean(axis=-1)).std()/(num_agents**.5)) for method in method_list}


    performance_list = [mean_performance_dict[m] for m in method_runtime_dict.keys()]
    performance_ci_list = [ci_performance_dict[m] for m in method_runtime_dict.keys()]
    relative_time_list = [mean_improvement_dict[m] for m in method_list]
    relative_time_ci_list = [ci_improvement_dict[m] for m in method_list]

    sr_list = [success_rates[m] for m in method_runtime_dict.keys()]#method_list]


    # plt.xticks(range(len(method_list)), method_list)
    plt.xticks(range(len(method_runtime_dict.keys())), list(method_runtime_dict.keys()))
    plt.xlabel("Method")
    plt.ylabel('Success rate')
    plt.title('Comparison of velocity target-setting methods')
    plt.bar(range(len(sr_list)), sr_list) 
    plt.savefig(os.path.join('results', "Method comparison -- Success Rate" + '.png'))
    plt.close()


    plt.xticks(range(len(method_runtime_dict.keys())), list(method_runtime_dict.keys()))
    plt.xlabel("Method")
    plt.ylabel('Time to complete')
    plt.title('Comparison of velocity target-setting methods')
    plt.bar(range(len(performance_list)), performance_list, yerr=performance_ci_list) 
    plt.savefig(os.path.join('results', "Method comparison -- Performance" + '.png'))
    plt.close()


    plt.xticks(range(len(method_list)), method_list)
    plt.xlabel("Method")
    plt.ylabel('Cost reduction over greedy baseline')
    plt.title('Comparison of velocity target-setting methods')
    plt.bar(range(len(relative_time_list)), relative_time_list, yerr=relative_time_ci_list) 
    plt.savefig(os.path.join('results', "Method comparison -- Relative Improvement" + '.png'))
    plt.close()




def get_random_search_costs(train_vel_agent, perm_search=True):
    pos_run_time_list = []
    vel_run_time_list = []
    # gd_step_list = [0,5,10]
    num_agents = NUM_AGENTS
    num_goals=NUM_GOALS
    n=N
    # gd_step_list = [0,1]
    # num_agents = 2
    # num_goals=2
    # n=2
    rand_time_list = []
    gd_time_list = []
    for _ in range(num_agents):
        vel_agent = train_vel_agent()
        rand_search_time_list = []
        gd_search_time_list = []
        for i in range(n):
            # goals = [np.random.rand(2)*2-1 for i in range(num_goals)]
            # pos = np.random.rand(2)*2-1
            goals = generate_path(num_goals + 1)
            pos = goals[0]
            goals = goals[1:-1]

            rand_test_time_list = []
            gd_test_time_list = []
            for gd_steps in gd_step_list:
                # min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.find_shortest_path(pos, goals, gd_steps=gd_steps, random_start=True, perm_search=perm_search)
                min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.find_shortest_path(pos, goals, gd_steps=gd_steps, random_start=False, perm_search=perm_search)
                print("GD: " + str(min_time))
                gd_test_time_list.append(len(min_trajectory))

                min_time, min_trajectory, min_path, min_vel_path, successful, pass_vals = vel_agent.find_shortest_path(pos, goals, gd_steps=gd_steps, random_search=True, perm_search=perm_search)
                print("random_search: " + str(min_time))
                rand_test_time_list.append(len(min_trajectory))
                
            rand_search_time_list.append(rand_test_time_list)
            gd_search_time_list.append(gd_test_time_list)
        
        rand_time_list.append(rand_search_time_list)
        gd_time_list.append(gd_search_time_list)
    
    rand_time_list = np.array(rand_time_list).squeeze()
    gd_time_list = np.array(gd_time_list).squeeze()
    # best = np.minimum(rand_time_list.min(axis=2),gd_time_list.min(axis=2))

    relative_time_change = (gd_time_list-rand_time_list)/rand_time_list
    relative_time_change = np.mean(relative_time_change, axis=1)

    # try:
    #     pickle.dump(vel_time_list, open("velocity_target.pkl", 'wb'))
    #     pickle.dump(pos_time_list, open("no_velocity_target.pkl", 'wb'))
    #     pickle.dump(relative_time_change, open("relative_time_change.pkl", 'wb'))
    # except:
    #     print("pickle failure")
    #     import pdb
    #     pdb.set_trace()

    mean = relative_time_change.mean(axis=0)
    ci = 2*relative_time_change.std(axis=0)/(num_agents**.5)
    steps = np.array(gd_step_list)

    plt.plot(steps, mean)
    plt.fill_between(steps, mean+ci, mean-ci, alpha=.4)
    plt.xlabel("Gradient steps")
    plt.ylabel("Relative Improvement vs random search")
    plt.title("Relative Improvement vs random search")
    plt.savefig(os.path.join('results', "Improvement vs random search" + '.png'))
    plt.close()


    t_score = stats.t.ppf(.975, num_agents)
    rands = rand_time_list.mean(axis=1)
    rand_mean = rands.mean(axis=0)
    rand_ci = t_score*rands.std(axis=0)/(num_agents**.5)

    gds = gd_time_list.mean(axis=1)
    gd_mean = gds.mean(axis=0)
    gd_ci = t_score*gds.std(axis=0)/(num_agents**.5)

    plt.plot(steps, rand_mean, color='red', label='Random Search')
    plt.fill_between(steps, rand_mean+rand_ci, rand_mean-rand_ci, alpha=.4, color='red')
    plt.plot(steps, gd_mean, color='blue', label='Gradient Descent')
    plt.fill_between(steps, gd_mean+gd_ci, gd_mean-gd_ci, alpha=.4, color='blue')
    plt.legend()
    plt.xlabel("Gradient steps")
    plt.ylabel("Relative Improvement vs random search")
    plt.title("Relative Improvement vs random search")
    plt.savefig(os.path.join('results', "Gradient Descent vs random search" + '.png'))
    plt.close()
