# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .constraint_abstract import *

class ConstraintInequality(ConstraintBase):
    def __init__(self, name, rows=None, cols=None, A=None, lb=None, ub=None):
        if rows is not None and cols is not None:
            ConstraintBase.__init__(self, name, rows=rows, cols=cols)
            self.lb = np.zeros(rows)
            self.ub = np.zeros(rows)
        elif A is not None and lb is not None and ub is not None:
            ConstraintBase.__init__(self, name, A=A)
            self.lb = copy.deepcopy(lb)
            self.ub = copy.deepcopy(ub)
            assert self.A.shape[0]== len(self.lb)
            assert self.A.shape[0]== len(self.ub)
        else:
            ConstraintBase.__init__(self, name)
    
    def getRows(self):
        assert self.A.shape[0]== len(self.lb)
        assert self.A.shape[0]== len(self.ub)
        return self.A.shape[0]

    def getCols(self):
        return self.A.shape[1]
    
    def resize(self, r, c):
        self.A = np.zeros((r, c))
        self.lb = np.zeros(r)
        self.ub = np.zeros(r)

    def isEquality(self):
        return False
    
    def isInequality(self):
        return True
            
    def isBound(self):
        return False

    def getVector(self):
        assert False
    
    def getLowerBound(self):
        return self.lb
    
    def getUpperBound(self):
        return self.ub
    
    def setVector(self, b):
        assert False

    def setLowerBound(self, lb):
        self.lb = copy.deepcopy(lb)

    def setUpperBound(self, ub):
        self.ub = copy.deepcopy(ub)

    def checkConstraint(self, x, tol):
        return np.all(np.dot(self.A, x) - self.ub <= tol) and np.all(np.dot(self.A, x) - self.lb >= -tol)

