# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator                         # Specified input files are called from here
from AirData import Atmosphere
from Blade_G import Blade                                       # For importing the relevant blade parameters like chord length, taper, etc.
import numpy as np

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
    
class simulation_data():
    # This class brings together all the force functions written above.
    def __init__(self):
        pass

    # Functions to generate forces data
    def generate_forces_x(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + collective_pitch) + lateral_pitch + 0.5 * longitudinal_pitch
        return x, y

    def generate_forces_y(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + tail_rotor_collective) + longitudinal_pitch - 0.3 * lateral_pitch
        return x, y

    def generate_forces_z(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.cos(tail_rotor_collective + lateral_pitch) + 0.2 * longitudinal_pitch
        return x, y

    def generate_forces_xyz(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        # Separate the components
        force_x = np.sin(x + collective_pitch)
        force_y = np.cos(x + tail_rotor_collective)
        force_z = np.sin(x) * np.cos(tail_rotor_collective + lateral_pitch) + 0.2 * longitudinal_pitch

        # Combine the individual components
        return x, force_x, force_y, force_z

    # Functions to generate moments data
    def generate_moments_x(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + collective_pitch) + np.sin(
            x) + 0.4 * lateral_pitch - 0.2 * longitudinal_pitch + tail_rotor_collective
        return x, y

    def generate_moments_y(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + collective_pitch - 0.7 * lateral_pitch + 0.5 * longitudinal_pitch
        return x, y

    def generate_moments_z(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + longitudinal_pitch) + collective_pitch - 0.6 * tail_rotor_collective + lateral_pitch
        return x, y

    def generate_moments_xyz(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        # Separate the components
        moment_x = np.cos(x + collective_pitch) + np.sin(x)
        moment_y = np.sin(x) + collective_pitch - 0.7 * lateral_pitch + 0.5 * longitudinal_pitch
        moment_z = np.cos(x + longitudinal_pitch) + collective_pitch - 0.6 * tail_rotor_collective + lateral_pitch

        # Combine the individual components
        return x, moment_x, moment_y, moment_z
