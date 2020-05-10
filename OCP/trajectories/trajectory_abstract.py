# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class TrajectorySample(object):
    def __init__(self, *args):
        self.pos = []
        self.vel = []
        self.acc = []

        if len(args) == 1:
            self.resize(args[0], args[0])
        elif len(args) == 2:
            self.resize(args[0], args[1])
        else:
            assert False, "The inputs of the class should be (size_pos and size_vel) or (size_pos only)"


    def resize(self, size_pos, size_vel):
        self.pos = np.zeros(size_pos)
        self.vel = np.zeros(size_vel)
        self.acc = np.zeros(size_vel)

    def setPos(self, pos):
        self.pos = copy.deepcopy(pos)

    def setVel(self, vel):
        self.vel = copy.deepcopy(vel)

    def setAcc(self, acc):
        self.acc = copy.deepcopy(acc)

    def getPos(self):
        return self.pos

    def getVel(self):
        return self.vel

    def getAcc(self):
        return self.acc


class TrajectoryBase(object):
    def __init__(self, name):
        self.name = name
        self.size = 0
        self.sample = []

    def getName(self):
        return self.name

    def size(self):
        return self.size

    def computeNext(self):
        return self.sample

    def getLastSample(self):
        return self.sample

    def has_trajectory_ended(self):
        return True
    
    