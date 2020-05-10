# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .task_motion import *
from ..constraints import ConstraintEquality

class TaskComEquality(TaskMotion):
    def __init__(self, name, robot):
        TaskMotion.__init__(self, name, robot)
        self.constraint = ConstraintEquality(name, rows=3, cols=robot.nv)
        self.Kp = np.zeros(3)
        self.Kd = np.zeros(3)
        self.p_error_vec = np.zeros(3)
        self.v_error_vec = np.zeros(3)
        self.p_com = np.zeros(3)
        self.v_com = np.zeros(3)
        self.a_des_vec = np.zeros(3)
        self.drift = np.zeros(3)

    def dim(self):
        return 3

    def getKp(self):
        return self.Kp

    def getKd(self):
        return self.Kd

    def setKp(self, Kp):
        self.Kp = copy.deepcopy(Kp)

    def setKd(self, Kd):
        self.Kd = copy.deepcopy(Kd)

    def setReference(self, ref):
        self.ref = copy.deepcopy(ref)

    def getReference(self):
        return self.ref
    
    def getDesiredAcceleration(self):
        return self.a_des_vec

    def getAcceleration(self, dv):
        return np.dot(self.constraint.matrix(), dv) -self.drift

    def position_error(self):
        return self.p_error_vec

    def velocity_error(self):
        return self.v_error_vec

    def position(self):
        return self.p_com

    def velocity(self):
        return self.v_com

    def position_ref(self):
        return self.ref.getPos()

    def velocity_ref(self):
        return self.ref.getVel()

    def getConstraint(self):
        return self.constraint

    def compute(self, time, q, v):
        self.p_com = self.robot.com(q, v)[0]
        self.v_com = self.robot.vcom(q, v)

        self.p_error_vec = self.p_com - self.ref.getPos()
        self.v_error_vec = self.v_com - self.ref.getVel()
        self.a_des_vec = -self.Kp * self.p_error_vec - self.Kd * self.v_error_vec + self.ref.getAcc()

        self.Jcom = self.robot.Jcom(q)
        self.constraint.setMatrix(self.Jcom)
        self.constraint.setVector(self.a_des_vec - self.drift)
        # print( self.ref.getPos())
        return self.constraint

