# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .constraint_abstract import *

class ConstraintEquality(ConstraintBase):
    def __init__(self, name, rows=None, cols=None, A=None, b=None):
        if rows is not None and cols is not None:
            ConstraintBase.__init__(self, name, rows=rows, cols=cols)
            self.b = np.zeros(rows)
        elif A is not None and b is not None:
            ConstraintBase.__init__(self, name, A=A)
            self.b = copy.deepcopy(b)
            assert self.A.shape[0]== len(self.b)
        else:
            ConstraintBase.__init__(self, name)
    
    def getRows(self):
        assert self.A.shape[0]== len(self.b)
        return self.A.shape[0]

    def getCols(self):
        return self.A.shape[1]
    
    def resize(self, r, c):
        self.A = np.zeros((r, c))
        self.b = np.zeros(r)

    def isEquality(self):
        return True
    
    def isInequality(self):
        return False
            
    def isBound(self):
        return False

    def getVector(self):
        return self.b
    
    def lowerBound(self):
        assert False
    
    def upperBound(self):
        assert False
    
    def setVector(self, b):
        self.b = copy.deepcopy(b)

    def setLowerBound(self, lb):
        assert False

    def setUpperBound(self, ub):
        assert False

    def checkConstraint(self, x, tol):
        return np.linalg.norm(np.dot(self.A, x) - self.b, 2) < tol

