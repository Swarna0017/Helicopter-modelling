# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator                         # Specified input files are called from here
from AirData import Atmosphere                                  # Fetches the required environmental sdata
from Blade_G import Blade 
from Airfoil import Airfoil_data                                      # For importing the relevant blade parameters like chord length, taper, etc.
import math

# All classes for implementing different hover performance prediction methods here
class BEMT_Implementer():
    def __init__(self, simulator_inputs=U_Inputs_Simulator, Blade=Blade, Atmosphere_data= Atmosphere, Airfoil=Airfoil_data):
                 
        self.A          = simulator_inputs.A
        self.rho        = Atmosphere_data.rho_calc()
        self.V          = simulator_inputs.V
        self.MRR        = simulator_inputs.MRR
        self.omega      = simulator_inputs.omega
        self.MR_nb      = simulator_inputs.MR_nb
        self.MR_omega   = (simulator_inputs.MR_omega)*math.pi*2/60
        self.chord_r    = Blade.chord()
        self.r          = Blade.Blade_sections(self, 10)
        self.Thrust     = 0                 # Initializing values
        self.Torque     = 0
        self.Power      = 0
        self.VW         = simulator_inputs.VW
        self.A          = simulator_inputs.MRA
        self.phi        = Airfoil.Phi()
        self.range      = simulator_inputs.No_of_iterations
        

        def solver(self):                                               # Defining a solver for calculating thrust, forces and moments
            self.v = -self.V*0.5 + math.sqrt((self.V*0.5)**2+(self.VW/(2*self.rho*self.A)))                 # Induced velocity




        

    
        


def MT_Implementer():
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
    
    # def BET_Implementer():
    

    # def BEMT_Implementor():
      
