# from OpenGL.GL import *
# from .geometric import Set
# from ..spaces.sets import Set
from ..spaces.objectives import TimeObjectiveFunction
from ..spaces.controlspace import LambdaKinodynamicSpace#, GymWrapperControlSpace

# from ..spaces.gymwrappers import GymWrapperControlSpace
from ..spaces.configurationspace import MultiConfigurationSpace, BoxConfigurationSpace
from ..spaces.sets import Set, BoxSet#
# from ..spaces.configurationspace import *
from ..spaces.so2space import SO2Space, so2
from ..spaces.biassets import BoxBiasSet
from ..planners.problem import PlanningProblem

# from ..spaces.edgechecker import *
# from ..spaces.metric import *
# from ..spaces.interpolators import *
# from ..spaces.geodesicspace import *
# from ..spaces.statespace import *
import random
import math

# class PendulumVisualizer:
#     def __init__(self,pendulum):
#         self.pendulum = pendulum

#     def drawObstaclesGL(self):
#         return

#     def toState(self,x,y):
#         return [(x*2*math.pi-math.pi)%(math.pi*2),y*(self.pendulum.omega_max-self.pendulum.omega_min)+self.pendulum.omega_min]

#     def toScreen(self,q):
#         return [((q[0]+math.pi)/math.pi*0.5)%1,(q[1]-self.pendulum.omega_min)/(self.pendulum.omega_max-self.pendulum.omega_min)]

#     def drawVerticesGL(self,qs):
#         glBegin(GL_POINTS)
#         for q in qs:
#             glVertex2f(*self.toScreen(q))
#         glEnd()

#     def drawRobotGL(self,q):
#         glColor3f(0,0,1)
#         glPointSize(7.0)
#         glBegin(GL_POINTS)
#         glVertex2f(*self.toScreen(q))
#         glEnd()

#     def drawGoalGL(self,goal):
#         bmin = self.toScreen(self.pendulum.goalmin)
#         bmax = self.toScreen(self.pendulum.goalmax)
#         glColor3f(1,0,0)
#         glLineWidth(2.0)
#         glBegin(GL_LINE_STRIP)
#         glVertex2f(1,bmin[1])
#         glVertex2f(bmin[0],bmin[1])
#         glVertex2f(bmin[0],bmax[1])
#         glVertex2f(1,bmax[1])
#         glEnd()
#         glBegin(GL_LINE_STRIP)
#         glVertex2f(0,bmax[1])
#         glVertex2f(bmax[0],bmax[1])
#         glVertex2f(bmax[0],bmin[1])
#         glVertex2f(0,bmin[1])
#         glEnd()
#         glLineWidth(1.0)

#     def drawInterpolatorGL(self,interpolator):
#         last = None
#         if isinstance(interpolator,PiecewiseLinearInterpolator):
#             glBegin(GL_LINE_STRIP)
#             for x in interpolator.path:
#                 v = self.toScreen(x)
#                 if last!=None:
#                     if (v[0] < 0.1 and last[0] > 0.9):
#                         glVertex2f(1,last[1])
#                         glEnd()
#                         glBegin(GL_LINE_STRIP)
#                         glVertex2f(0,v[1])
#                     elif (v[0] > 0.9 and last[0] < 0.1):
#                         glVertex2f(0,last[1])
#                         glEnd()
#                         glBegin(GL_LINE_STRIP)
#                         glVertex2f(1,v[1])
#                 glVertex2f(*v)
#                 last = v
#             glEnd()
#         else:
#             glBegin(GL_LINE_STRIP)
#             for s in range(10):
#                 u = float(s) / (9.0)
#                 x = interpolator.eval(u)
#                 v = self.toScreen(x)
#                 if last!=None:
#                     if (v[0] < 0.1 and last[0] > 0.9):
#                         glVertex2f(1,last[1])
#                         glEnd()
#                         glBegin(GL_LINE_STRIP)
#                         glVertex2f(0,v[1])
#                     elif (v[0] > 0.9 and last[0] < 0.1):
#                         glVertex2f(0,last[1])
#                         glEnd()
#                         glBegin(GL_LINE_STRIP)
#                         glVertex2f(1,v[1])
#                 glVertex2f(*v)
#                 last = v
#             glEnd()

class PendulumGoalSet(Set):
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
        self.c = [(a+b)/2 for a, b in zip(bmin, bmax)]
        self.box = BoxSet([0,bmin[1]],[so2.diff(self.bmax[0],self.bmin[0]),bmax[1]])
    def bounded(self):
        return True
    def sample(self):
        return [random.uniform(self.bmin[0],self.bmax[0]),random.uniform(self.bmin[1],self.bmax[1])]
    def contains(self,x):
        return self.box.contains([so2.diff(x[0],self.bmin[0]),x[1]])

    def signedDistance(self,x):
        return self.box.signedDistance([so2.diff(x[0],self.bmin[0]),x[1]])
    
    def signedDistance_gradient(self,x):
        return self.box.signedDistance_gradient([so2.diff(x[0],self.bmin[0]),x[1]])

def create_goal_region(center):
    etheta = 0.1
    eomega = 0.5
    # etheta = 0.00001
    # eomega = 0.00005
    goalmin = [center[0] - etheta, center[1] - eomega]
    goalmax = [center[0] + etheta, center[1] + eomega]
    return PendulumGoalSet(goalmin, goalmax)

class Pendulum:
    def __init__(self):
        self.theta_min = 0.0
        self.theta_max = 2*math.pi
        self.omega_min = -8#10.0
        self.omega_max = 8#10.0
        self.torque_min = -2.0
        self.torque_max = 2.0
        # self.torque_min = -1.
        # self.torque_max = 1.
        # etheta = 0.1
        # eomega = 0.5
        # self.goalmin = [math.pi-etheta,-eomega]
        # self.goalmax = [math.pi+etheta,+eomega]
        self.center = [math.pi, 0]
        self.bangbang = False
        # self.bangbang = True
        self.integrationStep = 0.01
        self.extendTime = 0.5
        self.g = 9.8
        self.m = 1
        self.L = 1

    def controlSpace(self):
        return LambdaKinodynamicSpace(self.configurationSpace(),self.controlSet(),self.derivative,self.integrationStep,self.extendTime)

    def controlSet(self):
        if self.bangbang:
            return BoxBiasSet([self.torque_min],[self.torque_max],concentration=float('inf'))
        else:
            return BoxSet([self.torque_min],[self.torque_max])

    def startState(self):
        return [0.0,0.0]

    def configurationSpace(self):
        res =  MultiConfigurationSpace(SO2Space(),BoxConfigurationSpace([self.omega_min],[self.omega_max]))
        #res.setDistanceWeights([1.0/(2*math.pi),1.0/(self.omega_max-self.omega_min)])
        return res

    def derivative(self,x,u):
        theta,omega = x
        return [omega,(u[0]/(self.m*self.L**2)-self.g*math.sin(theta)/self.L)]


    # def dynamics(self):
    #     return PendulumDynamics(self)
    
    # def goalSet(self):
    #     return PendulumGoalSet(self.goalmin,self.goalmax)
    def goalSet(self):
        return create_goal_region(self.center)





def pendulumTest():
    # gym_pendulum_test()
    p = Pendulum()
    objective = TimeObjectiveFunction()
    # return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
    #                        objective=objective,
    #                        visualizer=PendulumVisualizer(p), 
    #                        goal_maker=create_goal_region)
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective)
