# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator, Pilot_Inputs                         # Specified input files are called from here
from AirData import Atmosphere                                  # Fetches the required environmental sdata
from Blade_G import Blade 
from Airfoil import Airfoil_data                                      # For importing the relevant blade parameters like chord length, taper, etc.
from Inflow import v_calculator
import math
import numpy as np

# All classes for implementing different hover performance prediction methods here
class BEMT_Implementer():
    def __init__(self, simulator_inputs=U_Inputs_Simulator, pilot_inputs=Pilot_Inputs, Blade=Blade, Atmosphere_data= Atmosphere, Airfoil=Airfoil_data, v_data= v_calculator):
        self.rho_0              = 1.225             # kg/m^3   
        Atmosphere_data         = Atmosphere(simulator_inputs, pilot_inputs)
        Blade                   = Blade(simulator_inputs=simulator_inputs, pilot_inputs=pilot_inputs)     
        self.simulator_inputs   = simulator_inputs
        self.pilot_inputs       = Pilot_Inputs 
        self.rho                = Atmosphere_data.rho_calc()
        self.V                  = simulator_inputs.V
        self.MRR                = simulator_inputs.MRR
        self.MRA                = simulator_inputs.MRA
        self.omega              = simulator_inputs.MR_omega
        self.MR_nb              = simulator_inputs.MR_nb
        self.MR_rc              = simulator_inputs.MR_rc
        self.theta              = pilot_inputs.theta_0
        self.MR_omega           = (simulator_inputs.MR_omega)*math.pi*2/60
        self.VW                 = simulator_inputs.VW
        self.A                  = simulator_inputs.MRA
        self.r                  = Blade.Blade_sections(10)
        self.v                  = v_data.v_hover(self)  # Accessing v_hover from the instance
        self.chord_r            = Blade.chord()
        self.phi                = Airfoil.Phi(self)
        self.range              = simulator_inputs.Iterations
        self.v_data             = v_data(Blade, simulator_inputs)  # Creating an instance of v_calculator
        self.aoa                = Airfoil.AOA(self)  
        self.dr                 = Blade.dr
        self.MR_chord           = simulator_inputs.MR_chord


        self.Thrust, self.Torque, self.Power = self.BEMT_Solver()
        self.Ct, self.Cq, self.Cp = self.Coeff_finder(self.Thrust, self.Torque, self.Power)
        print(f"density: {self.rho}")
        print(f"T={self.Thrust}\nQ={self.Torque}\nP={self.Power}")
        print(f"Ct={self.Ct}\nCq={self.Cq}\nCp={self.Cp}")

    def Velocities(self,r):
        Ut = self.omega*r
        Up = self.V + self.V
        return Ut, Up
    
    def Prandtl_tip_loss_implemeter(self,r):
        f =  0.5*self.MR_nb*(self.MRR-r)*self.omega/(self.V+self.v)
        if(-f>500):
            F=1
        else:
            F=2*np.arccos(np.exp(-f))/np.pi
        return F
    
    def Coeff_finder(self, Thrust, Torque, Power):
        Ct = Thrust/(self.rho*self.MRA*(self.MR_omega*self.MRR)**2)
        Cq = Torque/(self.rho*self.MRR*self.MRA*(self.MR_omega*self.MRR)**2)
        Cp = Power/(self.rho*self.MRA*(self.MR_omega*self.MRR)**3)
        return Ct, Cq, Cp

    def BEMT_Solver(self):       # Defining a solver for calculating thrust, torque and power 
        self.Thrust     = 0                 # Initializing values
        self.Torque     = 0
        self.Power      = 0
        for i in range(len(self.r)):
            r               = self.r[i]
            chord           = np.array(self.chord_r[i])
            Ut, Up          = self.Velocities(r)
            phi             = np.array(self.phi[i])
            aoa             = self.aoa[i]
            cl, cd          = np.array(Airfoil_data.get_ClCd(self, aoa))
            F               = self.Prandtl_tip_loss_implemeter(r)
            dT              = 0.5*self.rho*(Ut**2+Up**2)*chord*(cl*np.cos(phi)-cd*np.sin(phi))*self.dr*F
            dQ              = 0.5*self.rho*(Ut**2+Up**2)*chord*(cl*np.cos(phi)-cd*np.sin(phi))*self.dr*F*r
            self.Thrust    += dT*self.MR_nb
            self.Torque    += dQ*self.MR_nb
        self.P=self.Torque*self.MR_omega
        return self.Thrust, self.Torque, self.Power
    
    
    



            
        



        

    
        


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
      
