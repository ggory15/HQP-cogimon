# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .constraint_abstract import *

class ConstraintBound(ConstraintBase):
    def __init__(self, name, size=None, lb=None, ub=None):
        if size is None and lb is None and ub is None:
            ConstraintBase.__init__(self, name)
        elif size is not None:
            ConstraintBase.__init__(self, name, A=np.identity(size))
            self.lb = np.zeros(size)
            self.ub = np.zeros(size)
        elif lb is not None and ub is not None:
            ConstraintBase.__init__(self, name, A=np.identity(len(lb)))
            self.lb = copy.deepcopy(lb)
            self.ub = copy.deepcopy(ub)
            assert len(self.lb) == len(self.ub)
    
    def rows(self):
        return len(self.lb)

    def cols(self):
        return len(self.lb)
    
    def resize(self, r, c):
        assert r == c
        self.A = np.identity(r)
        self.lb = np.zeros(r)
        self.ub = np.zeros(r)
    
    def isEquality(self):
        return False
    
    def isInequality(self):
        return False
            
    def isBound(self):
        return True

    def getVector(self):
        assert False
    
    def getLowerBound(self):
        return self.lb
    
    def getUpperBound(self):
        return self.ub
    
    def setVector(self):
        assert False

    def setLowerBound(self, lb):
        self.lb = copy.deepcopy(lb)

    def setUpperBound(self, ub):
        self.ub = copy.deepcopy(ub)

    def checkConstraint(self, x, tol):
        return np.all(x - self.ub <= tol) and np.all(x - self.lb >= -tol)


