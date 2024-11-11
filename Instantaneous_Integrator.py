# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator
from AirData import Atmosphere
from Blade_G import Blade

class Thrust():
    def __init__(self, simulator_inputs=U_Inputs_Simulator):
        Thrust=100