# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class ConstraintBase(object):
    def __init__(self, name, rows=None, cols=None, A=None):
        self.name = name
        if rows is not None:
            self.rows = rows
            self.cols = cols
            self.A = np.zeros((self.rows, self.cols))
        elif A is not None:
            self.A = A
    
    def getName(self):
        return self.name

    def getMatrix(self):
        return self.A
    
    def setMatrix(self, A):
        assert self.A.shape[0] == A.shape[0]
        assert self.A.shape[1] == A.shape[1]
        self.A = A
        return True

