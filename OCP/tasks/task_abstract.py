# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy

class TaskBase(object):
    def __init__(self, name, robot):
        self.name = name
        self.robot = robot
    
    def getName(self):
        return self.name
