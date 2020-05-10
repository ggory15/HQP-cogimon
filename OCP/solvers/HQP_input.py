# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class ConstraintLevel(object):
    def __init__(self, weight = None, constraint = None):
        if weight is None and constraint is None:
            self.constraints = []
        else:
            self.constraints = []
            self.setConstraint(weight, constraint)

    def setConstraint(self, weight, constraint):
        self.constraints.append([weight, constraint])
    
    def print(self, verbose=False):
        for i in range(len(self.constraints)):
            print (" -", end=" ")
            print("%r: " % (self.constraints[i][1].getName()), end=" ")
            print("w: ", self.constraints[i][0], end=" ")
            if self.constraints[i][1].isBound():
                print("bound ", len(self.constraints[i][1].getLowerBound()))
                if verbose:
                    print("lb: ", self.constraints[i][1].getLowerBound())
                    print("ub: ", self.constraints[i][1].getUpperBound())                   
            elif self.constraints[i][1].isEquality():
                print("equality", self.constraints[i][1].getRows(), " x ", self.constraints[i][1].getCols())
                if verbose:
                    print("A: ", self.constraints[i][1].getMatrix())
                    print("b: ", self.constraints[i][1].getVector())  
            elif self.constraints[i][1].isInequality():
                print("inequality", self.constraints[i][1].getRows(), " x ", self.constraints[i][1].getCols())
                if verbose:
                    print("A: ", self.constraints[i][1].getMatrix())
                    print("lb: ", self.constraints[i][1].getLowerBound())  
                    print("ub: ", self.constraints[i][1].getUpperBound())  
            else:
                assert False

    def getData(self):
        return self.constraints

    def eraseData(self, index):
        self.constraints[index] = None
        newconst = []
        for i in range(len(self.constraints)):
                if self.constraints[i] is not None:
                    newconst.append(self.constraints[i])
        self.constraints = newconst

class HQPData(object):
    def __init__(self):
        self.HQPData = []
        self.HQPData_arr = []
        self.level = 1

    def getData(self):        
        j = 0
        self.HQPData_arr = []
        if self.level == len(self.HQPData_arr):
            self.HQPData_arr = self.HQPData
        else:
            for i in range(len(self.HQPData)):
                if self.HQPData[i] is not None:
                    j += 1
                    self.HQPData_arr.append(self.HQPData[i])
            self.level = j
        return self.HQPData_arr

    def getSize(self):
        self.getData()
        return len(self.HQPData_arr)

    def setData(self, level, constraintLevel):
        while (level - len(self.HQPData) >= 0):
            self.HQPData.append(None)
        if self.HQPData[level] is None:
            self.HQPData[level] = constraintLevel
        else:
            for i in range(len(constraintLevel.getData())):
                self.HQPData[level].setConstraint(constraintLevel.getData()[i][0], constraintLevel.getData()[i][1])

    def eraseData(self, name):
        self.getData()
        for i in range(len(self.HQPData_arr)):
            cl = self.HQPData_arr[i]
            for j in range(0, len(cl.getData())):
                if cl.getData()[j][1].getName() == name:
                    cl.eraseData(j)
                    return True
        return False
    
    def print(self, verbose=False):
        level = 0
        for i in range(0, self.getSize()):
            print ("Level #", level, ":")
            self.HQPData_arr[i].print(verbose)
            level += 1

    # if verbose:
    #     level = 0
    #     for i in range(0, HQPData.size()):
    #         if HQPData.data()[i]:
    #             print "Level #", level, ":"
    #             for j in range(0, len(HQPData.data()[i])):
    #                 c = HQPData.data()[i][j][1]
    #                 if c.isEquality():
    #                     print "   Equality Task Name:", c.name
    #                     print "   A:", c.matrix()
    #                     print "   b", c.vector().transpose()
    #                 elif c.isInequality():
    #                     print "   Inequality Task Name:", c.name
    #                     print "   A:", c.matrix()
    #                     print "   lb", c.lowerBound().transpose()
    #                     print "   ub", c.upperBound().transpose()
    #                 else:
    #                     print "   lb", c.lowerBound().transpose()
    #                     print "   ub", c.upperBound().transpose()
    #             priority += 1 