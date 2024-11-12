# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator                         # Specified input files are called from here
from AirData import Atmosphere
from Blade_G import Blade                                       # For importing the relevant blade parameters like chord length, taper, etc.

# I have added the classes for implementing different hover performance prediction methods here
class MT_Implementor():
    def __init__(self) -> None:
        pass
        
class BET_Implementor():
    def __init__(self) -> None:
        pass

class BEMT_Implementor():
    def __init__(self) -> None:
        pass







class Thrust():
    def __init__(self, simulator_inputs=U_Inputs_Simulator):
        
        Atmosphere_Instance =  Atmosphere(simulator_inputs)     # Creating an instance of the atmosphere class to call the relevant functions
        self.A= simulator_inputs.A
        self.rho=Atmosphere_Instance.rho_calc()
        self.V=simulator_inputs.V
        self.MRR=simulator_inputs.MRR
        self.omega=simulator_inputs.omega
        self.v=100
        
    def Thrust_Calculator(self):
        Thrust_hover=2*self.v**2*self.rho*self.A                    # This calculates the Thrust for hover case (Momentum Theory)
        Thrust_Climb=2*self.rho*self.A*(self.v+self.V)              # This calculates the Thrust in case of climb (Momentum Theory)
        return Thrust_hover, Thrust_Climb
    
    def Power_Calculator(self):
        Thrust_hover, _=self.Thrust_Calculator()
        Power=Thrust_hover*self.v
        return Power

       
    def Coefficient_Calculator(self):
        Thrust_hover, _=self.Thrust_Calculator()
        Power=self.Power_Calculator()
        Ct=Thrust_hover/(self.rho*self.A*(self.omega*self.MRR)**2)
        Cp=Power/(self.rho*self.A*(self.omega*self.MRR)**3)
        return Ct, Cp