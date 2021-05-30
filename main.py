from __future__ import print_function,division
from six import iteritems

from pomp.planners import allplanners
from pomp.planners import test
# from pomp.example_problems import *
from pomp.example_problems import gym_pendulum
from pomp.example_problems import pendulum
# from pomp.example_problems import gym_pendulum_2 as gym_pendulum
from pomp.example_problems import gym_momentum
from pomp.example_problems import fetchrobot
# from pomp.example_problems import gym_car
from pomp.spaces.objectives import *

# from HER_mod.rl_modules.velocity_env import *
from train_distance_function import *

import time
import copy
import sys
import os,errno

numTrials = 30

def mkdir_p(path):
    """Quiet path making"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def testPlannerDefault(problem,problemName,maxTime,plannerType,**plannerParams):
    global numTrials
    print("Planning with",plannerType,'on problem',problemName)
    # planner = problem.planner(plannerType,**plannerParams)
    # prob_instance = problem()
    # planner = prob_instance.planner(plannerType,**plannerParams)
    folder = os.path.join("data",problemName)
    mkdir_p(folder)
    mkdir_p('demos/' + problemName + '_' + plannerType)
    # Orignal: 
        # test.testPlanner(planner,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'))

    #Variants
    test.testPlanner(problem,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'), plannerType, **plannerParams)
    # test.recordIters(problem,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'), plannerType, **plannerParams)
    # test.record_monitor(problem,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'), plannerType, problemName,  **plannerParams)
    # test.record_video_recorder(problem,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'), plannerType, problemName,  **plannerParams)
    # test.record_manual(problem,numTrials,maxTime,os.path.join(folder,allplanners.filename[plannerType]+'.csv'), plannerType, problemName,  **plannerParams)


# all_planners = ['ao-est','ao-rrt','r-est','r-est-prune','r-rrt','r-rrt-prune','rrt*','anytime-rrt','stable-sparse-rrt', 
#                 'rl-rrt', 'psst', 'gdsst', 'pgdsst', 'prlsst']
# rrt_planners = ['ao-rrt','anytime-rrt','r-rrt','r-rrt-prune','stable-sparse-rrt', 'rl-rrt', 'psst', 'gdsst', 'pgdsst', 'prlsst']
# est_planners = ['ao-est','r-est','r-est-prune']
all_planners = allplanners.all_planners
rrt_planners = allplanners.rrt_planners
est_planners = allplanners.est_planners

all_problems = {#'Kink':geometric.kinkTest(),
                # 'Bugtrap':geometric.bugtrapTest(),
                # 'Dubins':dubins.dubinsCarTest(),
                # 'Dubins2':dubins.dubinsTest2(),
                # 'Flappy':flappy.flappyTest(),
                # 'DoubleIntegrator':doubleintegrator.doubleIntegratorTest(),
                'Pendulum':pendulum.pendulumTest,
                # 'GymCar':gym_car.carTest(),
                'GymPendulum':gym_pendulum.gymPendulumTest,
                'GymMomentum':gym_momentum.gymMomentumTest,
                'FetchReach':fetchrobot.fetchReachTest,
                'FetchPush':fetchrobot.fetchPushTest,
                'FetchSlide':fetchrobot.fetchSlideTest,
                'FetchPickAndPlace':fetchrobot.fetchPickAndPlaceTest,
                #'LQR':lqr.lqrTest()
                }

# fetchrobotWitnessRadius = 0.0025#.01
fetchrobotWitnessRadius = .01#01
# fetchrobotWitnessRadius = .5#01
fetchSelectionRadius = 2*fetchrobotWitnessRadius
fetch_reach_time = 120
fetch_time = 600
# fetch_time = 30


defaultParameters = {'maxTime':30}
customParameters = {'Kink':{'maxTime':40,'nextStateSamplingRange':0.15},
                    'Bugtrap':{'maxTime':40,'nextStateSamplingRange':0.15},
                    'Pendulum':{'maxTime':1200,'edgeCheckTolerance':0.1,'selectionRadius':.3,'witnessRadius':0.16},
                    # 'Pendulum':{'maxTime':1200},#, 'pChooseGoal':0},
                    'GymPendulum':{'maxTime':30, 'edgeCheckTolerance':0.01,'selectionRadius':.3,'witnessRadius':0.16},
                    # 'GymPendulum':{'maxTime':7200},#,'selectionRadius':.03, 'witnessRadius':.01}, 
                    # 'GymMomentum':{'maxTime':30},#'selectionRadius':0.3,'witnessRadius':0.3},
                    'GymMomentum':{'maxTime':30,'selectionRadius':0.25,'witnessRadius':0.1},
                    'FetchReach':{'maxTime':fetch_reach_time,'witnessRadius':fetchrobotWitnessRadius,'selectionRadius':fetchSelectionRadius},
                    'FetchPush':{'maxTime':fetch_time},#,'witnessRadius':fetchrobotWitnessRadius,'selectionRadius':fetchSelectionRadius},
                    'FetchSlide':{'maxTime':fetch_time,'witnessRadius':fetchrobotWitnessRadius,'selectionRadius':fetchSelectionRadius},
                    'FetchPickAndPlace':{'maxTime':fetch_time,'witnessRadius':fetchrobotWitnessRadius,'selectionRadius':fetchSelectionRadius},
                    'Flappy':{'maxTime':120,'edgeCheckTolerance':4,'selectionRadius':70,'witnessRadius':35},
                    'DoubleIntegrator':{'maxTime':60,'selectionRadius':0.3,'witnessRadius':0.3},
                    'Dubins':{'selectionRadius':0.25,'witnessRadius':0.2},
                    'Dubins2':{'selectionRadius':0.25,'witnessRadius':0.2}
                    }

def parseParameters(problem,planner):
    global defaultParameters,customParameters
    params = copy.deepcopy(defaultParameters)
    if problem in customParameters:
        params.update(customParameters[problem])
    if '(' in planner:
        #parse out key=value,... string
        name,args = planner.split('(',1)
        if args[-1] != ')':
            raise ValueError("Planner string expression must have balanced parenthesis, i.e.: func ( arglist )")
        args = args[:-1]
        args = args.split(',')
        for arg in args:
            kv = arg.split("=")
            if len(kv) != 2:
                raise ValueError("Unable to parse argument "+arg)
            try:
                params[kv[0]] = int(kv[1])
            except ValueError:
                try:
                    params[kv[0]] = float(kv[1])
                except ValueError:
                    params[kv[0]] = kv[1]
        planner = name
    return planner,params

def runTests(problems = None,planners = None):
    global all_planners,all_problems
    if planners == None or planners == 'all' or planners[0] == 'all':
        planners = all_planners

    if problems == None or problems == 'all' or problems[0] == 'all':
        problems = all_problems.keys()

    for prname in problems:
        pr = all_problems[prname]
        for p in planners:
            p,params = parseParameters(prname,p)
            maxTime = params['maxTime']
            del params['maxTime']
            if pr().differentiallyConstrained() and p in allplanners.kinematicPlanners:
                #p does not support differentially constrained problems
                continue
            testPlannerDefault(pr,prname,maxTime,p,**params)
            print("Finished test on problem",prname,"with planner",p)
            print("Parameters:")
            for (k,v) in iteritems(params):
                print(" ",k,":",v)
    return

def runViz(problem,planner):
    #runVisualizer(rrtChallengeTest(),type=planner,nextStateSamplingRange=0.15,edgeCheckTolerance = 0.005)
    planner,params = parseParameters(problem,planner)
    if 'maxTime' in params:
        del params['maxTime']
    
    print("Planning on problem",problem,"with planner",planner)
    print("Parameters:")
    for (k,v) in iteritems(params):
        print(" ",k,":",v)
    runVisualizer(all_problems[problem],type=planner,**params)
    
if __name__=="__main__":
    #HACK: uncomment one of these to test manually
    #runViz('Kink','rrt*')
    #test KD-tree in noneuclidean spaces
    #runViz('Pendulum','ao-rrt(numControlSamples=10,nearestNeighborMethod=bruteforce)')
    #runViz('Pendulum','ao-rrt')
    #runViz('Dubins','stable-sparse-rrt(selectionRadius=0.25,witnessRadius=0.2)')
    #runViz('DoubleIntegrator','stable-sparse-rrt(selectionRadius=0.3,witnessRadius=0.3)')
    #runViz('Pendulum','stable-sparse-rrt(selectionRadius=0.3,witnessRadius=0.16)')
    #runViz('Flappy','stable-sparse-rrt(selectionRadius=70,witnessRadius=35)')

    if len(sys.argv) < 3:
        print("Usage: main.py [-v] Problem Planner1 ... Plannerk")
        print()
        print("  Problem can be one of:")
        print("   ",",\n    ".join(sorted(all_problems)))
        print("  or 'all' to test all problems.")
        print()
        print("  Planner can be one of:")
        print("   ",",\n    ".join(sorted(all_planners)))
        print("  or 'all' to test all planners.")
        print()
        print("  If -v is provided, runs an OpenGL visualization of planning")
        exit(0)
    if sys.argv[1] == '-v':
        from pomp.visualizer import runVisualizer
        #visualization mode
        print("Testing visualization with problem",sys.argv[2],"and planner",sys.argv[3])
        runViz(sys.argv[2],sys.argv[3])
    else:
        print()
        print("Testing problems",sys.argv[1],"with planners",sys.argv[2:])
        runTests(problems=[sys.argv[1]],planners=sys.argv[2:])
