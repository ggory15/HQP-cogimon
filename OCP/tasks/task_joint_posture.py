# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .task_motion import *
from ..constraints import ConstraintEquality

class TaskJointPosture(TaskMotion):
    def __init__(self, name, robot):
        TaskMotion.__init__(self, name, robot)
        if robot.nq == robot.nv:
            self.na = robot.model.nv
            self.freeflyer = False
        else:
            self.na = robot.model.nv - 6
            self.freeflyer = True
        
        self.ref = np.zeros(self.na)
        self.constraint = ConstraintEquality(name, rows=self.na, cols=robot.nv)
        self.Kp = np.zeros(self.na)
        self.Kd = np.zeros(self.na)
        self.m = np.ones(self.na)
        self.setMask(self.m)

    def getMask(self):
        return self.mask

    def setMask(self, m):
        assert len(self.m) == self.na
        self.mask = m
        dim = int(sum(m))
        S = np.zeros((dim, self.robot.nv))
        j = 0
        for i in range(dim):
            if m[i] is not 0:
                assert m[i] == 1
                S[j, self.robot.nv - self.na + i] = 1
                j += 1
        self.constraint.resize(dim, self.robot.nv)
        self.constraint.setMatrix(S)

    def dim(self):
        return int(sum(self.mask))

    def getKp(self):
        return self.Kp

    def getKd(self):
        return self.Kd

    def setKp(self, Kp):
        assert len(Kp) == self.na
        self.Kp = copy.deepcopy(Kp)

    def setKd(self, Kd):
        assert len(Kd) == self.na
        self.Kd = copy.deepcopy(Kd)

    def setReference(self, ref):
        assert self.na == len(ref.getPos())
        assert self.na == len(ref.getVel())
        assert self.na == len(ref.getAcc())
        self.ref = copy.deepcopy(ref)

    def getReference(self):
        return self.ref
    
    def getDesiredAcceleration(self):
        return self.a_des_vec

    def getAcceleration(self, dv):
        return np.dot(self.constraint.matrix(), dv)

    def position_error(self):
        return self.p_error_vec

    def velocity_error(self):
        return self.v_error_vec

    def position(self):
        return self.p

    def velocity(self):
        return self.v

    def position_ref(self):
        return self.ref.getPos()

    def velocity_ref(self):
        return self.ref.getVel()

    def getConstraint(self):
        return self.constraint

    def compute(self, time, q, v):
        if self.freeflyer:
            self.p = q[7:]
            self.v= v[6:]
        else:
            self.p = q
            self.v = v
        
        self.p_error_vec = self.p - self.ref.getPos()
        self.v_error_vec = self.v - self.ref.getVel()
        self.a_des_vec = -self.Kp * self.p_error_vec - self.Kd * self.v_error_vec + self.ref.getAcc()

        self.constraint.setVector(self.a_des_vec * self.mask)
        return self.constraint

