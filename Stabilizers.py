import math
import numpy as np
from U_inputs import U_Inputs_Simulator
from Blade_G import Blade
from AirData import Atmosphere
from Airfoil import Airfoil_data

class Stabilizer:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, blade: Blade, atmosphere: Atmosphere, airfoil: Airfoil_data):
        self.VS_chord = simulator_inputs.VS_chord
        self.VS_span  = simulator_inputs.VS_span
        self.VS_Cl, self.VS_Cd    = airfoil.get_ClCd()
        self.HS_chord = simulator_inputs.HS_chord
        self.HS_span  = simulator_inputs.HS_span
        self.HS_Cl, self.HS_Cd    = airfoil.get_ClCd()
        self.rho= atmosphere.rho_calc()
        self.VW       = simulator_inputs.VW
        self.Cd_body  = simulator_inputs.Cd_body
        self.area     = simulator_inputs.body_area
        self.Drag     = 0.5*self.Cd_body*self.rho*math.sqrt(self.Vf**2+self.V**2)*self.area
        self.aoa = np.arctan(self.Drag/self.VW)

        self.rho= atmosphere.rho_calc()
        self.V=simulator_inputs.V
        self.Vf=simulator_inputs.Vf
    
    def HS_Thrust(self):
        Thrust = 0.5*self.rho*math.sqrt(self.V**2+self.Vf**2)*self.HS_Cl*(self.HS_span*self.HS_chord)
        return Thrust
    def HS_Drag(self):
        Drag = 0.5*self.rho*math.sqrt(self.V**2+self.Vf**2)*self.HS_Cd*(self.HS_span*self.HS_chord)
        return Drag
    def VS_Thrust(self):
        Thrust = 0.5*self.rho*(self.V**2)*self.HS_Cl*(self.HS_span*self.HS_chord)
        return Thrust
    def VS_Drag(self):
        Drag = 0.5*self.rho*(self.V**2)*self.HS_Cd*(self.HS_span*self.HS_chord)
        return Drag
    