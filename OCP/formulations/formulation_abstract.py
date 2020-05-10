# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


class FormulationBase(object):
    def __init__(self, name, robot, verbose = False):
        self.name = name
        self.robot = robot
        self.verbose = verbose
        self.model = self.robot.model
        self.data = self.robot.data

    def nEq(self):
        return 0
    def nIn(self):
        return 0
    def nVar(self):
        return 0


    
    