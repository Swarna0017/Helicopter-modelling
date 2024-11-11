# This file calculates the environment data in which your helicopter is supposed to fly

import numpy as np
import math
from U_inputs import *
class Atmosphere():
    def __init__(self):
        self.alt=altitude