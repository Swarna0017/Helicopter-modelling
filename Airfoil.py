# Defines the airfoil profiles you use
# Different Airfoil models have been documented for usage. Selection as per necessity/benefit of usage.
import numpy as np
import math
from Blade_G import Blade
from AirData import Atmosphere
from U_inputs import U_Inputs_Simulator


class Airfoil_data:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, Atmosphere_data: Atmosphere):
        self.r      = Blade.Blade_sections(self, 10)
        self.V      = simulator_inputs.V
        self.omega  = simulator_inputs.MR_omega*2*np.pi*1/60            # omega = 2*pi*RPM/60
        self.rho    = Atmosphere_data.rho_calc()
        self.A      = simulator_inputs.MRA
        self.theta  = Blade.Pitch(self)
    
    def Phi(self,v):
        phi=np.arctan((self.V+v/self.omega.self.r))
        return phi
    
    def AOA(self, phi):
        aoa=np.radians(self.theta)-np.radians(phi)                       # Effective Angle of attack = theta - phi
        return aoa

    def get_ClCd(self, aoa):
        cl = 5.75*aoa
        cd = 0.013+1.25*aoa**2                  # Taking the sample/test case
        return cl, cd

    

