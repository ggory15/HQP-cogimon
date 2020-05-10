# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


class ContactBase(object):
    def __init__(self, name, robot):
        self.name = name
        self.robot = robot

    def name(self):
        return self.name
   

    
    