# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
from .trajectory_abstract import *

class TrajectoryEuclidianConstant(TrajectoryBase):
    def __init__(self, name, ref):
        TrajectoryBase.__init__(self, name)
        self.ref = ref
        self.setReference(ref)

    def size(self):
        return len(self.ref)

    def setReference(self, ref):
        self.sample = TrajectorySample(len(ref))
        self.sample.setPos(ref)

    def computeNext(self):
        return self.sample

    def getLastSample(self):
        return self.sample

    def getSample(self, time):
        return self.sample

    def has_trajectory_ended(self):
        return True
    
    