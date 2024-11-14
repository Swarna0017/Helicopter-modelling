# This file calculates the environment data in which your helicopter is supposed to fly

import numpy as np
import math
from U_inputs import U_Inputs_Simulator

# This class gives you the density, pressure, speed of sound at a particular altitude
class Atmosphere():
    
    def __init__(self, simulator_inputs: U_Inputs_Simulator, rho_0:float, T_0:float, P_0: float, Temp_grad: float):
        rho_0       = 1.225                                     # Standard air density MSL, kg/m^3
        T_0         = 298                                       # Standard temperature MSL, K
        P_0         = 101325                                    # Standard Pressure at MSL, Pa
        Temp_grad   = -6.5*10**(-3)                             # Slope upto 11 km considering max service ceiling of a helicopter, K/m
        self.alt=simulator_inputs.Altitude
        self.rho_0   = rho_0                                     
        self.T_0  = T_0 
        self.P_0     = P_0 
        self.Temp_grad = Temp_grad
        self.T1 = self.T_0+self.Temp_grad*self.alt

    def rho_calc(self):
        rho=self.rho_0*((self.T1/self.T_0)**4.2586)         # -(g/aR+1) ~ 4.2586 
        return rho
    
    def speed_of_sound_calc(self):
        speed_of_sound = math.sqrt(1.4*287*self.T1)         # Specific gas constant for air, R = 287 kJ/kg-K, Gamma=1.4 for air
        return speed_of_sound


