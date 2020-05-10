# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"

import pinocchio as se3
import numpy as np
from .contact_abstract import *

class Contact6d(ContactBase):
    def __init__(self, name, robot):
        ContactBase.__init__(self, name, robot)


   

    
    