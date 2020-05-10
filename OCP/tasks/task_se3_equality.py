# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import pinocchio as se3
import copy
from .task_motion import *
from ..constraints import ConstraintEquality
from ..trajectories import TrajectorySample

class TaskSE3Equality(TaskMotion):
    def __init__(self, name, robot, frameName):
        TaskMotion.__init__(self, name, robot)
        self.frameName = frameName
        self.constraint = ConstraintEquality(name, rows=6, cols=robot.nv)
        self.ref = TrajectorySample(12, 6)
        assert self.robot.model.existFrame(frameName)
        self.id = self.robot.model.getFrameId(frameName)

        self.v_ref = se3.Motion()
        self.a_Ref = se3.Motion()
        self.M_ref = se3.SE3()
        self.wMl = se3.SE3()
        self.p_error_vec = np.zeros(6)
        self.v_error_vec = np.zeros(6)
        self.p = np.zeros(12)
        self.v = np.zeros(6)
        self.p_ref = np.zeros(12)
        self.v_ref_vec = np.zeros(6)
        self.Kp = np.zeros(6)
        self.Kd = np.zeros(6)
        self.a_des = np.zeros(6)
        self.J = np.zeros((6, self.robot.nv))
        self.J_rotated = np.zeros((6, self.robot.nv))

        self.mask = np.ones(6)
        self.setMask(self.mask)
        self.local = True

    def getMask(self):
        return self.mask

    def setMask(self, m):
        self.mask = m
        n = self.dim()
        self.constraint.resize(n, self.J.shape[1])
        self.p_error_masked_vec = np.zeros(n)
        self.v_error_masked_vec = np.zeros(n)
        self.drift_masked = np.zeros(n)
        self.a_des_masked = np.zeros(n)

    def dim(self):
        return int(sum(self.mask))

    def getKp(self):
        return self.Kp

    def getKd(self):
        return self.Kd

    def setKp(self, Kp):
        assert len(Kp) == 6
        self.Kp = copy.deepcopy(Kp)

    def setKd(self, Kd):
        assert len(Kd) == 6
        self.Kd = copy.deepcopy(Kd)

    def setReference(self, ref):
        self.ref = copy.deepcopy(ref)
        self.M_ref.translation = copy.deepcopy(ref.getPos()[:3])
        self.M_ref.rotation = copy.deepcopy(  ref.getPos()[3:].reshape((3,3)))
        self.v_ref = se3.Motion(ref.getVel())
        self.a_ref = se3.Motion(ref.getAcc())

    def getReference(self):
        return self.ref
    
    def getDesiredAcceleration(self):
        return self.a_des_masked

    def getAcceleration(self, dv):
        return np.dot(self.constraint.matrix(), dv) + self.drift_masked

    def position_error(self):
        return self.p_error_masked_vec

    def velocity_error(self):
        return self.v_error_masked_vec

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

    def getFrameId(self):
        return self.id

    def useLocalFrame(self, local):
        self.local = local

    def compute(self, time, q, v):
        oMi = self.robot.framePlacement(q, self.id)
        v_frame = self.robot.frameVelocity(q, v, self.id)
        self.drift = self.robot.frameClassicAcceleration(self.id)

        self.J = self.robot.getFrameJacobian(self.id)
        self.p_error = se3.log(self.M_ref.inverse()* oMi)
        self.p_error_vec = self.p_error.vector
        self.p_ref = np.hstack((self.M_ref.translation, self.M_ref.rotation.flatten()))
        self.p = np.hstack((oMi.translation, oMi.rotation.flatten()))

        self.wMl.setIdentity()
        self.wMl.rotation = oMi.rotation

        if self.local:
            self.v_error = v_frame - self.wMl.actInv(self.v_ref)
            self.a_des = -self.Kp * self.p_error_vec - self.Kd * self.v_error.vector + self.wMl.actInv(self.a_ref).vector
        else:
            self.p_error_vec = np.dot(self.wMl.toActionMatrix() , self.p_error.vector)
            self.v_error = self.wMl.act(v_frame) - self.v_ref
            self.drift = self.wMl.act(self.drift)
            self.a_des = -self.Kp * self.p_error_vec - self.Kd * self.v_error.vector + self.a_ref.vector
            self.J_rotated = np.dot(self.wMl.toActionMatrix(), self.J)
            self.J = self.J_rotated

        self.v_error_vec = self.v_error.vector
        self.v_ref_vec = self.v_ref.vector
        self.v = v_frame.vector

        idx = 0
        for i in range(6):
            if np.abs(self.mask[i] - 1.0) < 1e-5:
                self.constraint.getMatrix()[idx, :] = self.J[i, :]
                self.constraint.getVector()[idx] = (self.a_des - self.drift.vector)[i]
                self.a_des_masked[i] = self.a_des[i]
                self.drift_masked[i] = self.drift.vector[i]
                self.p_error_masked_vec[i] = self.p_error_vec[i]
                self.v_error_masked_vec[i] = self.v_error_vec[i]
                
                idx += 1
   
        return self.constraint

