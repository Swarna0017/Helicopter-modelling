# Calculates required forces per rotation /azimuth circle
# i.e. This file is used for sectional load and forward flight thrust and power calculations
import numpy as np
import math
from AirData import Atmosphere
from U_inputs import *
from Inflow  import v_calculator
from Blade_G import Blade
from Airfoil import Airfoil_data

class Forward_flight_analyzer():
    def __init__(self, simulator_inputs: U_Inputs_Simulator, pilot_inputs: Pilot_Inputs, blade:Blade, Atmosphere_data: Atmosphere, Airfoil=Airfoil_data, v_data= v_calculator):
        self.rho_0      = 1.225             #kg/m^3   
        Atmosphere_data = Atmosphere(simulator_inputs, rho_0=1.225, T_0=298, P_0=101325, Temp_grad=-6.5e-3)
        Blade = Blade(simulator_inputs=simulator_inputs, pilot_inputs=pilot_inputs)     
        self.simulator_inputs=simulator_inputs
        self.pilot_inputs=Pilot_Inputs 
        self.rho        = Atmosphere_data.rho_calc()
        self.V          = simulator_inputs.V
        self.MRR        = simulator_inputs.MRR
        self.MRA        = simulator_inputs.MRA
        self.omega      = simulator_inputs.MR_omega
        self.MR_nb      = simulator_inputs.MR_nb
        self.MR_rc      = simulator_inputs.MR_rc
        self.MR_omega   = (simulator_inputs.MR_omega)*math.pi*2/60
        self.r          = Blade.Blade_sections(10)
        self.chord_r    = Blade.chord()
        self.VW         = simulator_inputs.VW
        self.A          = simulator_inputs.MRA
        self.phi        = Airfoil.Phi(self)
        self.range      = simulator_inputs.No_of_iterations
        self.v_data     = v_data(Blade, simulator_inputs)  # Creating an instance of v_calculator
        self.v          = self.v_data.v_hover()  # Accessing v_hover from the instance
        self.aoa        = Airfoil.AOA(self, self.phi)  
        self.dr         = Blade.dr
        self.MR_chord   = simulator_inputs.MR_chord
        self.Cd_body    = simulator_inputs.Cd_body
        self.body_area  = simulator_inputs.body_area

    def alpha_TPP_calc(self):
        Body_drag=self.Cd_body*0.5*self.rho*self.V*self.V*self.body_area
        alpha_tpp=np.arctan(Body_drag/self.VW)
        return (alpha_tpp)
    
    def Sectional_load_calc(self):




    