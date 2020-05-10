# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class HQPOutput(object):
    def __init__(self):
        self.x = []
        self.w = []
    
    def getOptVal(self):
        return self.x

    def getSlackVal(self):
        return self.w
