from __future__ import print_function,division
from six import iteritems
from builtins import range

from .profiler import Profiler
import time
import numpy as np
import copy
import gym
import types
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pyautogui, time
from ..spaces.plot_gd_sample import plot_traj, plot_roadmap
import pdb

# def testPlanner(planner,numTrials,maxTime,filename):    
def testPlanner(problem,numTrials,maxTime,filename, plannerType, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    successes = 0
    costs = []
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        planner = problem().planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        # while time.time()-t0 < maxTime:
        # while time.time()-t0 < maxTime and curCost == float('inf'):
        for _ in range(maxTime):
            planner.planMore(10)
            # try:
            #     planner.planMore(10)
            # except Exception as e:
            #     if hadException:
            #         print("Warning, planner raise two exceptions in a row. Quitting")
            #         break
            #     else:
            #         import traceback
            #         traceback.print_exc()
            #         print("Warning, planner raised an exception... soldiering on")
            #         print(e)
            #         hadException = True
            #         continue
            iters += 10
            if planner.bestPathCost != None and planner.bestPathCost != curCost:
                numupdates += 1
                curCost = planner.bestPathCost
                t1 = time.time()
                f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')
        if hasattr(planner,'stats'):
            print
            temp = Profiler()
            temp.items["Stats:"] = planner.stats
            temp.pretty_print()

        if curCost < float('inf'):
            costs.append(curCost)
            successes += 1

        print()
        print("Final cost:",curCost)
        print()

        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')

    total_cost = sum(costs)
    if successes > 0:
        ave_cost = total_cost/successes
    else: 
        ave_cost = float('inf')
    print('Success rate: ' + str(successes/numTrials))
    # print('Average cost: ' + str(ave_cost))
    print('Average cost: ' + str(np.mean(costs)))
    print('Cost STD: ' + str(np.std(costs)))
    print('Cost CI: ' + str(2*np.std(costs)/successes**.5))


    f.write('Num trials: ' + str(numTrials) + '\n')
    f.write('Success rate: ' + str(successes/numTrials) + '\n')
    f.write('Average cost: ' + str(np.mean(costs)) + '\n')
    f.write('Cost STD: ' + str(np.std(costs)) + '\n')
    f.write('Cost CI: ' + str(2*np.std(costs)/successes**.5) + '\n')


    f.close()



def recordIters(problem,numTrials,maxTime,filename, plannerType, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    filename = "gd_iter_test"
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    successes = 0
    trial=0
    t1 = time.time()

    print()
    print("Trial",trial+1)# 
    planner = problem().planner(plannerType,**plannerParams)
    planner.reset()
    curCost = float('inf')
    t0 = time.time()
    numupdates = 0
    iters = 0
    hadException = False
    while time.time()-t0 < maxTime:
    # while time.time()-t0 < maxTime and curCost == float('inf'):
        planner.planMore(100)
        iters += 100
        f.write(str(trial)+","+str(iters)+","+str(t1-t0)+'\n')
        t1 = time.time()

    if curCost < float('inf'):
        successes += 1

    print()
    print("Final cost:",curCost)
    print()

    f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()

    print('Success rate: ' + str(successes/numTrials))



def record_video_recorder(problem,numTrials,maxTime,filename, plannerType, problemName, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        problem_instance = problem()
        planner = problem_instance.planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        while time.time()-t0 < maxTime:
            planner.planMore(10)
            iters += 10
            if planner.bestPathCost != None and planner.bestPathCost != curCost:
                numupdates += 1
                curCost = planner.bestPathCost
                t1 = time.time()
                f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')
        if hasattr(planner,'stats'):
            print
            temp = Profiler()
            temp.items["Stats:"] = planner.stats
            temp.pretty_print()
        print()
        print("Final cost:",curCost)
        print()

        controlSpace = problem_instance.controlSpace


        env = controlSpace.env
        goal = env.goal
        # env = Monitor(controlSpace.env, './demos/' + problemName + '_' + plannerType + '/')

        recorder = VideoRecorder(controlSpace.env, base_path='./demos/' + problemName + '_' + plannerType, 
            enabled=True)


        #horrible hack to fix a bug in gym's video recorder (present on May 8th, 2021)
        #Please remove if no longer needed
        frame = env.render("rgb_array")
        recorder.encoder = gym.wrappers.monitoring.video_recorder.ImageEncoder(recorder.path, frame.shape, recorder.frames_per_sec, recorder.output_frames_per_sec)
        def capture_frame(self, frame):
            if not isinstance(frame, (np.ndarray, np.generic)):
                raise gym.error.InvalidFrame('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
            if frame.shape != self.frame_shape:
                raise gym.error.InvalidFrame("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(frame.shape, self.frame_shape))
            if frame.dtype != np.uint8:
                raise gym.error.InvalidFrame("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))
            self.proc.stdin.write(frame.tobytes())
        recorder.encoder.capture_frame = types.MethodType(capture_frame, recorder.encoder)


        path = planner.getPath()
        initial_state = np.array(path[0][0])
        env.set_state(env, initial_state)

        # env = controlSpace.env
        # env.goal = 
        control_sequence = []
        for mini_control_seq in path[1]:
            for control in mini_control_seq: 
                control_sequence.append(np.array(control))

        import pdb
        env.render()
        recorder.capture_frame()
        obs2 = initial_state.tolist()
        epsilon = .01
        reward = controlSpace.goal_contains(obs2)



        for control in control_sequence: 
            obs1, reward, done, info = env.step(control)
            env.render()
            recorder.capture_frame()
            print(reward)

        recorder.close()
        recorder.enabled = False
        env.close()

        # pdb.set_trace()

        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()



def record_monitor(problem,numTrials,maxTime,filename, plannerType, problemName, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        problem_instance = problem()
        planner = problem_instance.planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        while time.time()-t0 < maxTime:
            planner.planMore(10)
            iters += 10
            if planner.bestPathCost != None and planner.bestPathCost != curCost:
                numupdates += 1
                curCost = planner.bestPathCost
                t1 = time.time()
                f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')
        if hasattr(planner,'stats'):
            print
            temp = Profiler()
            temp.items["Stats:"] = planner.stats
            temp.pretty_print()
        print()
        print("Final cost:",curCost)
        print()

        controlSpace = problem_instance.controlSpace
        import pdb


        env = controlSpace.env
        goal = env.goal
        env = Monitor(controlSpace.env, './demos/' + problemName + '_' + plannerType, force=True)
        env.reset()

        # pdb.set_trace()
        env.env.goal = goal



        frame = env.render("rgb_array")
        recorder = env.video_recorder
        recorder.encoder = gym.wrappers.monitoring.video_recorder.ImageEncoder(recorder.path, frame.shape, recorder.frames_per_sec, recorder.output_frames_per_sec)
        def capture_frame(self, frame):
            if not isinstance(frame, (np.ndarray, np.generic)):
                raise gym.error.InvalidFrame('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
            if frame.shape != self.frame_shape:
                raise gym.error.InvalidFrame("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(frame.shape, self.frame_shape))
            if frame.dtype != np.uint8:
                raise gym.error.InvalidFrame("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))
            self.proc.stdin.write(frame.tobytes())
        recorder.encoder.capture_frame = types.MethodType(capture_frame, recorder.encoder)

        # recorder = VideoRecorder(controlSpace.env, base_path='./demos/' + problemName + '_' + plannerType)

        path = planner.getPath()
        initial_state = np.array(path[0][0])
        env.set_state(env, initial_state)

        # env = controlSpace.env
        # env.goal = 
        control_sequence = []
        for mini_control_seq in path[1]:
            for control in mini_control_seq: 
                control_sequence.append(np.array(control))

        env.render()
        # recorder.capture_frame()
        obs2 = initial_state.tolist()
        epsilon = .01
        reward = controlSpace.goal_contains(obs2)



        for control in control_sequence: 
            obs1, reward, done, info = env.step(control)
            env.render()
            print(reward)

        env.close()
        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()


def record_manual(problem,numTrials,maxTime,filename, plannerType, problemName, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        problem_instance = problem()
        planner = problem_instance.planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        while time.time()-t0 < maxTime:
            planner.planMore(10)
            iters += 10
            if planner.bestPathCost != None and planner.bestPathCost != curCost:
                numupdates += 1
                curCost = planner.bestPathCost
                t1 = time.time()
                f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')
        if hasattr(planner,'stats'):
            print
            temp = Profiler()
            temp.items["Stats:"] = planner.stats
            temp.pretty_print()
        print()
        print("Final cost:",curCost)
        print()

        controlSpace = problem_instance.controlSpace


        env = controlSpace.env
        goal = env.goal

        path = planner.getPath()
        initial_state = np.array(path[0][0])

        control_sequence = []
        for mini_control_seq in path[1]:
            for control in mini_control_seq: 
                control_sequence.append(np.array(control))

        import pdb
        obs2 = initial_state.tolist()
        epsilon = .01
        reward = controlSpace.goal_contains(obs2)
        env.render()
        # pdb.set_trace()

        i=0
        file_loc = './demos/' + problemName + '_' + plannerType + '/'
        screenshot = pyautogui.screenshot()
        screenshot.save(file_loc + str(i) + '.png')

        for control in control_sequence: 
            i+=1
            # time.sleep(.5)
            obs1, reward, done, info = env.step(control)
            env.render()
            # pdb.set_trace()
            # time.sleep(.5)
            screenshot = pyautogui.screenshot()
            screenshot.save(file_loc + str(i) + '.png')
            print(reward)

        # recorder.close()
        # recorder.enabled = False
        pdb.set_trace()
        env.close()

        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()



def path_visualization_2D(problem,numTrials,maxTime,filename, plannerType, problemName, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    # f = open(filename,'w')
    # f.write("trial,plan iters,plan time,best cost\n")
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        problem_instance = problem()
        planner = problem_instance.planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        # while time.time()-t0 < maxTime:
        for _ in range(maxTime):
        # while time.time()-t0 < maxTime and curCost == float('inf'):
            planner.planMore(10)
            iters += 10
            if planner.bestPathCost != None and planner.bestPathCost != curCost:
                numupdates += 1
                curCost = planner.bestPathCost
                t1 = time.time()
                # f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')
        # if hasattr(planner,'stats'):
        #     print
        #     temp = Profiler()
        #     temp.items["Stats:"] = planner.stats
        #     temp.pretty_print()
        print()
        print("Final cost:",curCost)
        print()

        controlSpace = problem_instance.controlSpace


        env = controlSpace.env
        goal = env.goal

        path = planner.getPath()
        roadmap = planner.getRoadmap()
        # initial_state = np.array(path[0][0])
        initial_state = np.array(roadmap[0][0])
        env.set_state(env, initial_state)

        control_sequence = []
        if path != None: 
            for mini_control_seq in path[1]:
                for control in mini_control_seq: 
                    control_sequence.append(np.array(control))

        import pdb
        obs2 = initial_state.tolist()
        epsilon = .01
        reward = controlSpace.goal_contains(obs2)
        # env.render()
        # pdb.set_trace()

        i=0
        # file_loc = './demos/' + problemName + '_' + plannerType + '/'
        # screenshot = pyautogui.screenshot()
        # screenshot.save(file_loc + str(i) + '.png')
        # xs = [obs2[0]]
        # ys = [obs2[1]]
        traj = [obs2]
        for control in control_sequence: 
            i+=1
            obs1, reward, done, info = env.step(control)
            traj.append(obs1['observation'])

        traj_list = []
        # pdb.set_trace()
        for i in range(len(roadmap[1])):
            head = roadmap[1][i][0]
            tail = roadmap[1][i][1]
            state = roadmap[0][head]
            env.set_state(env, state)
            current_traj = [state]
            for control in roadmap[1][i][2]:
                assert np.array(control).shape == (2,)
                obs1, reward, done, info = env.step(control)
                current_traj.append(obs1['observation'])
            current_traj.append(roadmap[0][tail])

            traj_list.append(current_traj)

        plot_roadmap(obs2, traj_list, traj, goal)

        # plot_traj(obs2, traj, goal)
        # recorder.close()
        # recorder.enabled = False
        # pdb.set_trace()
        env.close()

    #     f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    # f.close()
