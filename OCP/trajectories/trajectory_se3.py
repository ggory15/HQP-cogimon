# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
from .trajectory_abstract import *

class TrajectorySE3Constant(TrajectoryBase):
    def __init__(self, name, M):
        TrajectoryBase.__init__(self, name)
        self.M = M
        self.setReference(M)
        
    def size(self):
        return len(6)

    def setReference(self, M):
        self.sample = TrajectorySample(12, 6)
        M_ref_vec = np.hstack((M.translation, M.rotation.flatten()))
        self.sample.setPos(M_ref_vec)

    def computeNext(self):
        return self.sample

    def getLastSample(self):
        return self.sample

    def getSample(self, time):
        return self.sample

    def has_trajectory_ended(self):
        return True
    
    