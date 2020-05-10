# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .task_motion import *
from ..constraints import ConstraintInequality

class TaskJointBounds(TaskMotion):
    def __init__(self, name, robot):
        TaskMotion.__init__(self, name, robot)
        if robot.nq == robot.nv:
            self.na = robot.model.nv
            self.freeflyer = False
        else:
            self.na = robot.model.nv - 6
            self.freeflyer = True
        
        self.velocity_limit = 2000.0
        self.lb = self.velocity_limit * np.ones(robot.nv)
        self.ub = -1.0 * self.lb
        self.constraint = ConstraintInequality(name, A=np.identity(robot.nv), lb=self.lb, ub =self.ub)
        self.Kp = np.zeros(self.na)
        self.Kd = np.zeros(self.na)
        self.m = np.ones(self.na)
        self.setMask(self.m)
        self.buffer = 0.01

    def getMask(self):
        return self.mask

    def setMask(self, m):
        self.mask = m
        n = self.dim()
        self.constraint.resize(n, self.robot.nv)
        
    def dim(self):
        return int(sum(self.mask))

    def getKp(self):
        return self.Kp

    def getKd(self):
        return self.Kd

    def setKp(self, Kp):
        assert len(Kp) == self.robot.nv
        self.Kp = copy.deepcopy(Kp)

    def setKd(self, Kd):
        assert len(Kd) == self.robot.nv
        self.Kd = copy.deepcopy(Kd)

    def setReference(self, lb, ub):
        assert len(self.lb) == len(lb)
        assert len(self.ub) == len(ub)
        self.lb = lb
        self.ub = ub

    def getReference(self):
        return self.lb, self.ub

    def getAcceleration(self, dv):
        return np.dot(self.constraint.getMatrix(), dv) 

    def getConstraint(self):
        return self.constraint

    def compute(self, time, q, v):
        idx = 0
        for i in range(self.robot.nv):            
            if np.abs(self.mask[i] - 1.0) < 1e-5:
                self.constraint.getMatrix()[0, i] = 1.0
                if q[i] < self.lb[i] + self.buffer:
                    self.constraint.getLowerBound()[idx] = self.Kp[i] * (self.lb[i] + self.buffer - q[i]) - self.Kd[i] * v[i]
                    self.constraint.getUpperBound()[idx] = self.velocity_limit
                    if self.constraint.getLowerBound()[idx] > self.velocity_limit:
                        self.constraint.getLowerBound()[idx] = self.velocity_limit
                elif q[i] > self.ub[i] - self.buffer:
                    self.constraint.getUpperBound()[idx] = self.Kp[i] * (self.ub[i] - self.buffer - q[i]) - self.Kd[i] * v[i]
                    self.constraint.getLowerBound()[idx] = -self.velocity_limit
                    if self.constraint.getUpperBound()[idx] < -self.velocity_limit:
                        self.constraint.getUpperBound()[idx] = -self.velocity_limit
                else:
                    self.constraint.getUpperBound()[idx] = self.velocity_limit
                    self.constraint.getLowerBound()[idx] = -self.velocity_limit
    
                idx += 1
        return self.constraint

