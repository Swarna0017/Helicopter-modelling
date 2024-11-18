# Calculates required forces per rotation /azimuth circle
# i.e. This file is used for sectional load and forward flight thrust and power calculations
import numpy as np
import math
from AirData import Atmosphere
from U_inputs import *
from Inflow  import v_calculator
from Blade_G import Blade
from Airfoil import Airfoil_data
from Instantaneous_Integrator import BEMT_Implementer, Forward_flight_analyzer


    
import numpy as np
from U_inputs import U_Inputs_Simulator, Pilot_Inputs
from AirData import Atmosphere


class Cyclic_analyzer:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, pilot_inputs: Pilot_Inputs):
        """
        Initializes the rotor dynamics class with input data and atmospheric conditions.
        """
        # Inputs from U_inputs and AirData
        self.simulator_inputs = simulator_inputs
        self.pilot_inputs = pilot_inputs
        self.atmosphere = Atmosphere(simulator_inputs, pilot_inputs)
        self.V = simulator_inputs.V
        self.MR_theta=10
        self.R = simulator_inputs.MRR
        self.r = np.linspace(simulator_inputs.MR_rc, simulator_inputs.MRR, 100)
        self.MRR = simulator_inputs.MRR
        self.MR_rc =  simulator_inputs.MR_rc
        self.MR_root_twist = simulator_inputs.MR_root_twist
        self.MR_tip_twist = simulator_inputs.MR_tip_twist
        self.MR_theta = Blade.Pitch(self)
        self.omega = simulator_inputs.MR_omega * np.pi * 2 / 60  # Convert RPM to rad/s
        self.alpha_tpp = Forward_flight_analyzer.alpha_TPP_calc(self)
        self.dr = (self.r[1] - self.r[0])  # Blade section size
        self.c = np.full(100, simulator_inputs.MR_chord)  # Chord length at each section
        self.b = simulator_inputs.MR_nb  # Number of blades
        self.I = simulator_inputs.VW / (self.omega**2)  # Moment of inertia
        self.rho = self.atmosphere.rho_calc()  # Atmospheric density


        
        # self.theta_0 = pilot_inputs.theta_0
        # self.theta_1s = pilot_inputs.theta_1s
        # self.theta_1c = pilot_inputs.theta_1c

    def calculate_vertical_forces(self, thrust_req, theta_0, theta_1s, theta_1c):
        """
        Computes forces and moments acting on the rotor blades during vertical flight.
        """
        T_sections = np.zeros((100, 100))
        D_sections = np.zeros((100, 100))
        roll_moment, pitch_moment, beta, total_torque, total_thrust = 0, 0, 0, 0, 0

        # Discretize azimuthal angles
        azimuth_angles = np.linspace(0, 2 * np.pi, 100)
        d_psi = azimuth_angles[1] - azimuth_angles[0]
        azimuth_angles += d_psi / 2

        # Compute sectional forces for each azimuthal angle
        for i, psi in enumerate(azimuth_angles):
            T_sections[i], D_sections[i] = self.calculate_vertical_section_forces(self,
                thrust_req, psi, theta_0, theta_1s, theta_1c)

        # Integrate forces and moments over azimuth and blade sections
        for i, psi in enumerate(azimuth_angles):
            for j, r_j in enumerate(self.r):
                roll_moment += T_sections[i][j] * r_j * np.sin(psi) * self.dr * d_psi
                pitch_moment -= T_sections[i][j] * r_j * np.cos(psi) * self.dr * d_psi
                beta += T_sections[i][j] * r_j * self.dr * d_psi
                total_torque += D_sections[i][j] * r_j * self.dr * d_psi
                total_thrust += T_sections[i][j] * self.dr * d_psi

        # Scale results
        roll_moment *= self.b / (2 * np.pi)
        pitch_moment *= self.b / (2 * np.pi)
        beta *= 1 / (2 * np.pi * self.I * self.omega**2)
        total_torque *= self.b / (2 * np.pi)
        total_thrust *= self.b / (2 * np.pi)

        return thrust_req, total_thrust, total_torque, roll_moment, pitch_moment, beta

    def calculate_forward_forces(self, thrust_req, alpha_TPP, theta_0, theta_1s, theta_1c):
        """
        Computes forces and moments acting on the rotor blades during forward flight.
        """
        T_sections = np.zeros((100, 100))
        D_sections = np.zeros((100, 100))
        roll_moment, pitch_moment, beta, total_torque, total_thrust = 0, 0, 0, 0, 0

        # Discretize azimuthal angles
        azimuth_angles = np.linspace(0, 2 * np.pi, 100)
        d_psi = azimuth_angles[1] - azimuth_angles[0]
        azimuth_angles += d_psi / 2

        # Compute sectional forces for each azimuthal angle
        for i, psi in enumerate(azimuth_angles):
            T_sections[i], D_sections[i] = self.calculate_forward_section_forces(
                thrust_req, alpha_TPP, psi, theta_0, theta_1s, theta_1c
            )

        # Integrate forces and moments over azimuth and blade sections
        for i, psi in enumerate(azimuth_angles):
            for j, r_j in enumerate(self.r):
                roll_moment += T_sections[i][j] * r_j * np.sin(psi) * self.dr * d_psi
                pitch_moment -= T_sections[i][j] * r_j * np.cos(psi) * self.dr * d_psi
                beta += T_sections[i][j] * r_j * self.dr * d_psi
                total_torque += D_sections[i][j] * r_j * self.dr * d_psi
                total_thrust += T_sections[i][j] * self.dr * d_psi

        # Scale results
        roll_moment *= self.b / (2 * np.pi)
        pitch_moment *= self.b / (2 * np.pi)
        beta *= 1 / (2 * np.pi * self.I * self.omega**2)
        total_torque *= self.b / (2 * np.pi)
        total_thrust *= self.b / (2 * np.pi)

        return thrust_req, total_thrust, total_torque, roll_moment, pitch_moment, beta

    def calculate_vertical_section_forces(self, thrust_req, psi, r, theta_0, theta_1s, theta_1c):
        theta_root = theta_0 + theta_1s*np.sin(psi) + theta_1c*np.cos(psi)
        self.dr                 =  (self.MRR-self.MR_rc)/10
        theta = Blade.Pitch(self)
        #calculating induced velocity for the given climb speed and blade dimensions
        lambda_forward = v_calculator.lambda_forward(psi, r, self.alpha_tpp, T)
        v_i = lambda_forward*self.MRR*self.omega
        #calculating Perpendicular and Tangential Components
        U_P = self.V*np.sin(self.alpha_tpp) + v_i + self.V*np.cos(psi)*np.sin(self.beta_0)
        U_T = self.omega*r + self.V*np.cos(self.alpha_tpp)*np.sin(psi)
        cl, cd = Airfoil_data.get_ClCd()
        chord_r = Blade.chord()
        #calculating Angle of Attack at each section
        phi = np.arctan2(U_P, U_T)
        alpha_eff = theta - phi
        #arrays to store sectional Thrust and Drag
        T = []
        D = []
        #Assuming stall after angles <-15 and >15 degrees
        for i in range(np.size(alpha_eff)):
            self.aoa=alpha_eff
            if (alpha_eff[i]>-15*np.pi/180 and alpha_eff[i]<15*np.pi/180):
                cl, cd = Airfoil_data.get_ClCd()
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
            else:
                cl = 0
                cd = self.Cd0
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
        T = np.array(T)
        D = np.array(D)
        #returning the given sectional forces array                 
        return T, D
    
    def calculate_forward_section_forces(self, thrust_req, alpha_TPP, psi, r, theta_0, theta_1s, theta_1c):
        theta_root = theta_0 + theta_1s*np.sin(psi) + theta_1c*np.cos(psi)
        self.dr                 = Blade.dr
        theta = Blade.Pitch()
        #calculating induced velocity for the given climb speed and blade dimensions
        lambda_forward = v_calculator.lambda_forward(psi, r, self.alpha_tpp, T)
        v_i = lambda_forward*self.MRR*self.omega
        #calculating Perpendicular and Tangential Components
        U_P = self.V*np.sin(self.alpha_tpp) + v_i + self.V*np.cos(psi)*np.sin(self.beta_0)
        U_T = self.omega*r + self.V*np.cos(self.alpha_tpp)*np.sin(psi)
        cl, cd = Airfoil_data.get_ClCd()
        chord_r = Blade.chord()
        #calculating Angle of Attack at each section
        phi = np.arctan2(U_P, U_T)
        alpha_eff = theta - phi
        #arrays to store sectional Thrust and Drag
        T = []
        D = []
        #Assuming stall after angles <-15 and >15 degrees
        for i in range(np.size(alpha_eff)):
            self.aoa=alpha_eff
            if (alpha_eff[i]>-15*np.pi/180 and alpha_eff[i]<15*np.pi/180):
                cl, cd = Airfoil_data.get_ClCd()
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
            else:
                cl = 0
                cd = self.Cd0
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
        T = np.array(T)
        D = np.array(D)
        #returning the given sectional forces array                 
        return T, D
    
