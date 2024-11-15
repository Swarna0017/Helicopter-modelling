# Calculates inflow ratios for different flight regimes
from U_inputs import U_Inputs_Simulator
from Blade_G import Blade
from AirData import Atmosphere
import math

class v_calculator:
    def __init__(self, Blade: Blade, simulator_inputs: U_Inputs_Simulator):
        self.V      = simulator_inputs.V
        self.omega  = simulator_inputs.MR_omega*2/60*math.pi
        self.VW     = simulator_inputs.VW


    def v_hover(self):
        v = -self.V*0.5 + math.sqrt((self.V*0.5)**2+(self.VW/(2*self.rho*self.A)))                 # Induced velocity for hover, Momentum theory
        return v
        