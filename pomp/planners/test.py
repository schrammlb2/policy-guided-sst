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

# def testPlanner(planner,numTrials,maxTime,filename):    
def testPlanner(problem,numTrials,maxTime,filename, plannerType, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    f = open(filename,'w')
    f.write("trial,plan iters,plan time,best cost\n")
    successes = 0
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
        while time.time()-t0 < maxTime and curCost == float('inf'):
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
            successes += 1

        print()
        print("Final cost:",curCost)
        print()

        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()

    print('Success rate: ' + str(successes/numTrials))



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
    # while time.time()-t0 < maxTime:
    while time.time()-t0 < maxTime and curCost == float('inf'):
    # for i in range(50):
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
        # pdb.set_trace()
        obs2 = initial_state.tolist()
        epsilon = .01
        reward = controlSpace.goal_contains(obs2)
        # env.render()

        i=0
        file_loc = './demos/' + problemName + '_' + plannerType + '/'
        screenshot = pyautogui.screenshot()
        # screenshot.save(file_loc + str(i) + '.png')

        point_path = [obs2]
        for control in control_sequence: 
            i+=1
            # time.sleep(.5)
            obs1, reward, done, info = env.step(control)
            point_path.append(obs1['observation'].tolist())
            # env.render()
            # pdb.set_trace()
            # time.sleep(.5)
            screenshot = pyautogui.screenshot()
            # screenshot.save(file_loc + str(i) + '.png')
            print(reward)

        n = 1000
        n_gd_samples = [controlSpace.configuration_sampler.sample() for _ in range(n)]
        n_no_gd_samples = [controlSpace.configuration_sampler.configurationSpace.sample() for _ in range(n)]
        dist = lambda x, y: ((x-y)**2).sum()**.5
        # gd_dists = [ min([dist(np.array(samp), np.array(point)) for point in point_path]) for samp in n_gd_samples]
        # no_gd_dists = [ min([dist(np.array(samp), np.array(point)) for point in point_path]) for samp in n_no_gd_samples]
        gd_dists = [ dist(np.array(samp), np.array(point_path[-1])) for samp in n_gd_samples]
        no_gd_dists = [ dist(np.array(samp), np.array(point_path[-1])) for samp in n_no_gd_samples]
        gd_mean = sum(gd_dists)/n
        no_gd_mean = sum(no_gd_dists)/n
        # recorder.close()
        # recorder.enabled = False
        # pdb.set_trace()
        print("gd_mean: " + str(gd_mean))
        print("no_gd_mean: " + str(no_gd_mean))
        env.close()

        f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')
    f.close()
