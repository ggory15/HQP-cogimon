# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .task_abstract import *

class TaskMotion(TaskBase):
    def __init__(self, name, robot):
        TaskBase.__init__(self, name, robot)
        self.mask = 0 

    def setMask(self, mask):
        self.mask = copy.deepcopy(mask)

    def hasMase(self):
        return self.mask is not 0


