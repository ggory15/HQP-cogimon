# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class HQPBase(object):
    def __init__(self, name):
        self.name = name
        self.maxIter = 1000
        self.maxTime = 100.0
        self.useWarmstart = True
    
    def setmaxIter(self, maxIter):
        self.maxIter = copy.deepcopy(maxIter)

    def setmaxTime(self, maxTime):
        self.maxTime = copy.deepcopy(maxTime)


