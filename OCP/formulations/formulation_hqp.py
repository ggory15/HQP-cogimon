# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"

from .formulation_abstract import *
from ..constraints import ConstraintEquality, ConstraintInequality, ConstraintBound
from ..solvers import HQPData, ConstraintLevel
import numpy as np
import copy
import pinocchio as se3

class TaskLevel(object):
    def __init__(self, task, priority):
        self.task = task
        self.priority = priority

class ContactLevel(object):
    def __init__(self, contact):
        self.contact = contact

class FormulationHQP(FormulationBase):
    def __init__(self, name, robot, verbose = False):
        FormulationBase.__init__(self, name, robot, verbose)
        self.baseDynamics = ConstraintEquality("baseDyanmics", rows=6, cols=robot.nv )
        self.solutionDecoded = False
        self.t = 0.0
        self.v = robot.nv

        self.freeflyer = False
        if robot.nq is not robot.nv:
            self.u = 6
            self.freeflyer = True
        else:
            self.u = 0

        self.k = 0
        self.eq = copy.deepcopy(self.u)
        self.ineq = 0
        self.bound = 0
        self.hqpData = HQPData()
        self.Jc = np.zeros((self.k, self.v))
        if self.freeflyer:
            self.hqpData.setData(0, ConstraintLevel(1.0, self.baseDynamics))
        self.taskMotions = []

    def getnVar(self):
        return self.v + self.k
    
    def getnEq(self):
        return self.eq

    def getnIn(self):
        return self.ineq

    def resizeHQPdata(self):
        self.Jc = np.zeros((self.k, self.v))
        self.baseDynamics.resize(self.u, self.v + self.k)
        for i in range(0, self.hqpData.getSize()):
            cl = self.hqpData.getData()[i]
            for j in range(0, len(cl.getData())):
                c = cl.getData()[j][1]
                c.resize(c.getRows(), self.v + self.k)

    def addTask(self, tl, weight, priorityLevel):
        c = tl.task.getConstraint()
        if c.isEquality():
            tl.constraint = ConstraintEquality(c.getName(), rows=c.getRows(), cols=self.v + self.k)
            if priorityLevel == 0:
                self.eq += c.getRows()
        elif c.isInequality():
            tl.constraint = ConstraintInequality(c.getName(), rows=c.getRows(), cols=self.v + self.k)
            if priorityLevel == 0:
                self.ineq += c.getRows()
        else:
            tl.constraint = ConstraintBound(c.getName(), size=self.v + self.k)
            if priorityLevel == 0:
                self.bound += len(c.getLowerBound())
        self.hqpData.setData(priorityLevel, ConstraintLevel(weight, tl.task.getConstraint()))

    def addMotionTask(self, task, weight, priorityLevel, transition_duration = 0.0):
        assert (weight >= 0.0)
        assert (transition_duration >= 0.0)

        tl = TaskLevel(task, priorityLevel)
        self.taskMotions.append(tl)
        self.addTask(tl, weight, priorityLevel)
        return True

    def updateTaskWeight(self, name, weight):
        for i in range(self.hqpData.getSize()):
            cl = self.hqpData.getData()[i]
            for j in range(0, len(cl.getData())):
                if cl.getData()[j][1].getname() == name:
                    cl.getData()[j][0] = weight
                    return True    
        return False

    def computeProblemData(self, time, q, v):
        se3.computeAllTerms(self.model, self.data, q, v)
        
        if self.freeflyer:
            M_a = self.data.M[self.u:, :]
            h_a = self.robot.nle(q, v)[self.u:]
            M_u = self.data.M[:self.u, :]
            h_u = self.robot.nle( q, v)[:self.u]

            # self.baseDynamics.setMatrix(np.hstack((M_u, -J_u)))
            # self.baseDynamics.setVector(-h_u)
        else:
            self.M_a = self.data.M
            self.h_a = self.robot.nle(q, v)
            for i in range(0, len(self.taskMotions)):
                c = self.taskMotions[i].task.compute(time, q, v)
                
        self.solutionDecoded = False
        return self.hqpData

    def decodeSolution(self, sol):
        if (self.solutionDecoded):
            return True
        if self.freeflyer:
            print ("here")
        else:
            self.dv = sol[len(sol)-1].x
            self.tau = self.h_a + np.dot(self.M_a, self.dv)

    def getAccelerations(self, sol):
        self.decodeSolution(sol)
        return self.dv

    def removeFromHQPdata(self, name):
        found = self.hqpData.eraseData(name)
        return found

    def removeMotionTask(self, name):
        assert self.removeFromHQPdata(name)
        for i in range(len(self.taskMotions)):
            if self.taskMotions[i].task.getName() == name:
                if (self.taskMotions[i].priority == 0):
                    if self.taskMotions[i].task.getConstraint().isEquality():
                        self.eq -= self.taskMotions[i].task.getConstraint().rows()
                    elif self.taskMotions[i].task.getConstraint().isInequality():
                        self.ineq -= self.taskMotions[i].task.getConstraint().rows()
                    else:
                        self.bound -= len(self.taskMotions[i].task.getConstraint().getLowerBound())
                np.delete(self.taskMotions, i, 0)
                return True
        return False


