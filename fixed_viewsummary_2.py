
#!/usr/bin/env python
from __future__ import print_function,division
from six import iteritems
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict
import pdb

if len(sys.argv) < 3:
    print("Usage: viewsummary.py csvfile item")
    exit(0)

successFraction = 0.5

labelmap = {"lazy_rrgstar":"Lazy-RRG*",
            "lazy_birrgstar":"Lazy-BiRRG*",
            "prmstar":"PRM*",
            "fmmstar":"FMM*",
            "lazy_prmstar":"Lazy-PRM*",
            "lazy_rrgstar_subopt_0.1":"Lazy LBT-RRG*, eps=0.1",
            "lazy_rrgstar_subopt_0.2":"Lazy LBT-RRG*, eps=0.2",
            "fmtstar":"FMT*",
            #"rrtstar":"RRT*",
            "birrtstar":"RRT*",
            "rrtstar_subopt_0.1":"LBT-RRT*(0.1)",
            "rrtstar_subopt_0.2":"LBT-RRT*(0.2)",
            "policy_guided_gradient_descent_sst": "PSST (proposed)",
            # "policy_guided_sst": "PSST without GD Sampling",
            "policy_guided_sst": "PSST (proposed)",
            "stable_sparse_rrt": "Stable Sparse RRT (SST)",
            "rl": "RL (SAC + HER)",
            "rl_rrt": "RL-RRT",
            "rl-then-sst": "Best of (RL, SST)"
}
#labelorder = ["restart_rrt_shortcut","prmstar","fmmstar","rrtstar","birrtstar","rrtstar_subopt_0.1","rrtstar_subopt_0.2","lazy_prmstar","lazy_rrgstar","lazy_birrgstar"]
labelorder = ["ao_rrt","ao_est","repeated_rrt","repeated_est","repeated_rrt_prune","repeated_est_prune","stable_sparse_rrt","anytime_rrt","rrtstar"]
labelorder += ['gradient_descent_sst', 'policy_guided_gradient_descent_sst', 'policy_guided_sst', 'rl', 'rl_rrt', 'rl-then-sst']
dashes = [[],[8,8],[4,4],[2,2],[1,1],[12,6],[4,2,2,2],[8,2,2,2,2,2],[6,2],[2,6]]
ylabelmap = {"best cost":"Path length",
             "numEdgeChecks":"# edge checks",
             "success fraction ci": "Success rate",
             "best cost ci": "Best cost"
}

# ignored_labels = ['gradient_descent_sst']
ignored_labels = []
# ignored_labels += ['policy_guided_sst']
ignored_labels += ['policy_guided_gradient_descent_sst']
ignored_labels += ['stable_sparse_rrt']
ignored_labels += ['rl']
ignored_labels += ['rl-rrt']
# ignored_labels += ['rl-then-sst']
# ignored_labels = ['gradient_descent_sst', 'stable_sparse_rrt', 'rl']
# labelmap['policy_guided_gradient_descent_sst'] = 'PSST with gradient descent'
# labelmap['policy_guided_sst'] = 'PSST without gradient descent'
ignored_labels += ['gradient_descent_sst']



timevarname = 'time'
# timevarname = 'numIters'
#timevarname = 'numMilestones'
item = sys.argv[2]
# if item == 'bestCost':
#     item = 'best cost mean'
#     # item = 'best cost'
# if item == 'successFraction':
#     item = 'success fraction'
stat_list = ['mean', 'std', 'min', 'max', 'ci', '']

with open(sys.argv[1],'r') as f:
    reader = csv.DictReader(f)
    items = defaultdict(list)
    for row in reader:
        time = dict()
        vmean = dict()
        # vstd = dict()
        # vmin = dict()
        # vmax = dict()
        # vci  = dict()
        vdict = {stat: {} for stat in stat_list}
        skip = dict()
        # for (k,v) in row.iteritems():
        for (k,v) in row.items():
            if k[-2:] == 'ci': 
                tup = v[1: -1].split(',')
                v = [float(t) for t in tup] if len(v) > 0 else None
            else: 
                v = float(v) if len(v) > 0 else None
            words = k.split(None,1)
            label = words[0]
            if len(words) >= 2 and words[1] == timevarname:
                time[label] = v

            if item == 'best cost' and len(words) >= 2 and words[1] == 'success fraction':
                if type(v) == type(None) or v < successFraction:
                    skip[label] = True
                else:
                    skip[label] = False
            if len(words) >= 2 and words[1].startswith(item):
                suffix = words[1][len(item)+1:]
                if suffix not in stat_list:
                    print("Warning, unknown suffix",suffix)
                    continue

                if suffix=='':
                    suffix = 'mean'
                else: 
                    pass
                    # pdb.set_trace()
                vdict[suffix][label] = v
                # vdict['mean'] = {label: v}
        
        # for label,t in time.iteritems():
        for label,t in time.items():
            # if label in skip and skip[label]:
            #     pdb.set_trace()
            #     items[label].append((t,None))
            # elif label in vmean:
            #     items[label].append((t,vmean[label]))
            if label in vdict['mean']:
                items[label].append((t,vdict['mean'][label]))
            else:
                print("Warning, no item",item,"for planner",label,"read")
        # pdb.set_trace()
    print("Available planners:",list(items.keys()))

    #small, good for printing
    #fig = plt.figure(figsize=(4,2.7))
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    # if timevarname=='time':
    #     ax1.set_xlabel("Time (s)")
    # else:
    #     ax1.set_xlabel("Iterations")
    ax1.set_xlabel("Iterations")
    minx = 0
    maxx = 0
    ax1.set_ylabel(ylabelmap.get(item,item))
    # x_list = [items[label][-1][0] for label in labelorder if (
    #     label in items and 
    #     label not in ignored_labels and 
    #     items[label][-1][0] is not None)]
    # maximum_x = max(x_list)
    forced_max = 1000#250
    # forced_max = float('inf')
    for n,label in enumerate(labelorder):
        if label not in items: continue
        if label in ignored_labels: continue
        plot = items[label]
        plot = [elem for elem in items[label] if elem[0] is not None and elem[0] < forced_max]
        # plot.append([maximum_x, plot[-1][1]])
        if len(items[label])==0:
            print("Skipping item",label)
        x,y = zip(*plot)
        if label == 'rl': 
            y = [y[-1]]*len(y)
        minx = min(minx,*[v for v in x if v is not None])
        maxx = max(maxx,*[v for v in x if v is not None])
        plannername = labelmap[label] if label in labelmap else label
        print("Plotting",plannername)
        if item[-2:] != 'ci': 
            line = ax1.plot(x,y,label=plannername)#,dashes=dashes[n])
            plt.setp(line,linewidth=1.5)
        else:
            for i in y: 
                if i != None:
                    assert i[0] < float('inf') and i[1] < float('inf')
            # y_plus =  [(i if i==None else i[0]) for i in y]
            # y_minus = [(i if i==None else i[1]) for i in y]
            y_minus = [i[0] for i in y if i is not None]
            y_plus =  [i[1] for i in y if i is not None]
            y_mean =  [i[2] for i in y if i is not None]

            x = [i for i in x if i is not None]
            plt.fill_between(x, y_plus, y_minus, alpha=.4)
            line = ax1.plot(x,y_mean,label=plannername)#,dashes=dashes[n])
            plt.setp(line,linewidth=1.5)
    #plt.legend(loc='upper right');
    plt.title("Simple 2D -- Success rate")
    # plt.title("FetchPickAndPlace -- Success rate")
    titlemap = {"success fraction ci": "Success rate",
             "best cost ci": "Cost"
    }
    env_name = sys.argv[1].split('/')[1]
    if "Gym" in env_name: 
    	env_name = env_name[3:]
    # plt.title("FetchReach -- " + titlemap[item])

    plt.title( env_name + " -- " + titlemap[item])
    # plt.title("Asteroids w/ Distribution Shift -- " + titlemap[item])
    # plt.title("Asteroids -- " + titlemap[item])
    # plt.title("FetchReach -- Success rate")
    # plt.title("Asteroids w/ Distribution Shift -- Cost")
    plt.legend();
    #good for bugtrap cost
    #plt.ylim([2,3])
    #good for other cost
    #plt.ylim([1,2])
    #good for edge checks
    if item=="numEdgeChecks":
        plt.ylim([0,800])
    else:
        #plt.ylim([2,2.8])
        pass
    if timevarname=='time':
        if sys.argv[1].startswith('tx90'):
            plt.xlim([0,20])
        elif sys.argv[1].startswith('baxter'):
            plt.xlim([0,60])
        elif sys.argv[1].startswith('bar_25'):
            plt.xlim([0,20])
        else:
            plt.xlim([math.floor(minx),math.ceil(maxx)])
    else:
        plt.xlim([0,5000])

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend
    #frame = legend.get_frame()
    #frame.set_facecolor('0.97')

    # Set the fontsize
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
    #      ncol=3, fancybox=True, columnspacing=0, handletextpad=0, shadow=False)
    #for label in legend.get_texts():
    #    label.set_fontsize(9)
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0, box.width, box.height* 0.8])
    #plt.setp(ax1.get_xticklabels(),fontsize=12)
    #plt.setp(ax1.get_yticklabels(),fontsize=12)
    #start,end = ax1.get_ylim()
    #ax1.yaxis.set_ticks(np.arange(start, end, 0.1))

    plt.show()
