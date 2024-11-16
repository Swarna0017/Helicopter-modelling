# mission_planner.py
import math
import numpy as np
import matplotlib.pyplot as plt

from Blade_G    import Blade
from AirData    import Atmosphere
from U_inputs   import U_Inputs_Simulator, U_Inputs_Planner
from Inflow     import v_calculator


  # Available power MSL, kW

class Hover_Climb():
    def __init__(self, simulator_inputs: U_Inputs_Simulator, mission_inputs: U_Inputs_Planner, atmosphere: Atmosphere, blade: Blade):
        self.VW         = mission_inputs.VW
        self.FW         = mission_inputs.FW
        self.MR_chord   = simulator_inputs.MR_chord
        self.MRR        = simulator_inputs.MRR
        self.T_0        = 288
        self.Altitude   = simulator_inputs.Altitude
        self.rho        = atmosphere.rho_calc()
        self.rho_0      = 1.225 # MSL, kg/m^3
        self.solidity   = blade.Disk_solidity(self)
        self.omega      = simulator_inputs.MR_omega
        self.Cd         = 0.03
        self.MRA        = simulator_inputs.MRA
        self.SFC        = mission_inputs.SFC
        self.hover_time = 0.000000001
        self.omega      = (30*simulator_inputs.MR_omega)/math.pi
        self.Temp_grad  = -0.0065
        self.P_sea_level = 300

    
    def rho_finder(self,h):
        T1 = self.T_0+self.Temp_grad*h
        rho= self.rho_0*(T1/self.T_0)**4.2586
        return rho

    def Performance(self, rho, h, VW):
        rho             = self.rho_finder(h)
        P_available     = self.P_sea_level * (rho / self.rho_0)
        P_prof          = (rho * self.solidity * self.Cd * ((self.omega * self.MRR) ** 3) * math.pi * (self.MRR ** 2)) / 8  # W
        P_induced       = math.sqrt((VW ** 3) / (2 * rho * self.MRA)) # W
        P_R             = P_induced + P_prof  # in Watts

        # Calculate Rate of Climb (RC)
        RC              = (P_available*1000 - P_R) / self.VW  # in m/s

        # Calculate endurance (hours)
        fuel_flow_rate  = self.SFC * P_R  # in kg/s
        endurance       = self.FW / (fuel_flow_rate)  # in seconds

        # Check if fuel available is sufficient for the mission
        # if self.hover_time > endurance.any():
        #     raise ValueError("Mission failed: Insufficient fuel")
        
        return P_available, P_induced, P_prof, P_R, endurance, RC


    def Power_Outputs(self, simulator_inputs, mission_inputs, atmosphere, blade):
        # Create an instance of Hover_Climb
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        rho=atmosphere.rho_calc()

        # Call the Hover_and_Climb_Performance function
        P_available, P_induced, P_prof, P_R, endurance, RC = self.Performance(self.rho, self.Altitude, self.VW)

        # Output mission results
        print(f"Induced Power (kW): {P_induced / 1000:.3f}")
        print(f"Profile Power (kW): {P_prof / 1000:.3f}")
        print(f"Total Power required (kW): {P_R / 1000:.3f}")
        print(f"Power Available (kW): {P_available:.3f}")
        print(f"Endurance (seconds): {endurance:.3f}")
        print(f"Maximum Rate of Climb (m/s): {RC:.3f}")

        return P_available, P_induced, P_prof, P_R, endurance, RC

    def Power_vs_Alt(self, simulator_inputs, mission_inputs, atmosphere, blade):
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        altitudes = np.linspace(0, 12000, 100)  # Altitude variation in meters

        # Lists to store values for plotting
        PI      = []
        P_pr    = []
        PR      = []
        PA      = []
        r_c     = []

        for h in altitudes:
            rho                 = self.rho_finder(h) 
        
            P_available, P_induced, P_prof, P_R, endurance, RC = self.Performance(rho, h, self.VW)
            PI.append(P_induced/1000)
            P_pr.append(P_prof/1000)
            PR.append(P_R/1000)
            PA.append(P_available)
            r_c.append(RC)

            # Output ceiling information
            if -0.001 <= RC <= 0.001:
                print(f"Absolute ceiling (m): {h:.3f}")
            elif 0.507 <= RC <= 0.509:
                print(f"Service ceiling (m): {h:.3f}")

        plt.plot(altitudes,  PI, label='Induced Power (kW)', color='green')
        plt.plot(altitudes, P_pr, label='Profile Power (kW)', color='orange')
        plt.plot(altitudes, PR, label='Power Required (kW)', color='black')
        plt.plot(altitudes, PA, label='Power Available (kW)', color='blue')
        plt.title('Power vs Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid()
        plt.xlim(0, 8000)
        return plt.show()

    def RC_vs_Alt(self, simulator_inputs, mission_inputs, atmosphere, blade):
        r_c = []
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)

        altitudes   = np.linspace(0, 12000, 100)

        for h in altitudes:
            rho                 = self.rho_finder(h) 
            _, _, _, _, _, RC   = self.Performance(rho, h, self.VW)
            r_c.append(RC)
        # Rate of Climb vs Altitude
        plt.plot(altitudes, r_c, label='Rate of Climb (m/s)', color='blue')
        plt.title('R/C vs Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('Rate of Climb (m/s)')
        plt.legend()
        plt.grid()
        plt.xlim(0, 8000)
        plt.ylim(0, 1000)
        return plt.show()


    # Mission planner for multiple take-off weights (plotting)
    def RC_vs_weight(self, simulator_inputs, mission_inputs, atmosphere, blade):
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        # List to store climb rates
        r_c = []
        VW = np.linspace(1, 400 * 9.81, 1000)  # Take-off weight variation in Newtons
        rho=atmosphere.rho_calc()

        for W in VW:
            P_available, P_induced, P_prof, P_R, endurance, RC = self.Performance(self.rho, self.Altitude, W)
            r_c.append(RC)

        # Plot Rate of Climb vs Take-off weight
        plt.plot(VW, r_c, label='Rate of Climb (m/s)', color='blue')
        plt.title('Rate of Climb vs Gross weight')
        plt.xlabel('Gross weight (N)')
        plt.ylabel('Rate of Climb (m/s)')
        plt.legend()
        plt.grid()
        plt.xlim(1, 400 * 9.81)
        return plt.show()

class Forward_Flight_Performance:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, mission_inputs: U_Inputs_Planner, atmosphere: Atmosphere, blade: Blade):
        # Extract simulator and mission parameters
        self.VW             = simulator_inputs.VW
        self.FW             = simulator_inputs.FW
        self.Altitude       = simulator_inputs.Altitude
        self.SFC            = mission_inputs.SFC
        self.MRR            = simulator_inputs.MRR
        self.MRA            = simulator_inputs.MRA
        self.rpm            = simulator_inputs.rpm
        self.solidity       = blade.Disk_solidity()
        self.omega          = simulator_inputs.MR_omega
        self.C_d            = simulator_inputs.Blade_Cd
        self.C_L            = simulator_inputs.Blade.Cl
        self.Vf             = simulator_inputs.Vf
        self.distance       = simulator_inputs.distance
        self.D              = simulator_inputs.D
        self.power_loss     = mission_inputs.installed_power_loss
        self.rho_0          = 1.225              # Sea level air density in kg/m^3
        self.g              = 9.81               # Gravitational acceleration in m/s^2
        self.R0             = 287.05             # Specific gas constant for air in J/(kg*K)
        self.T0             = 288.15             # Sea level standard temperature in K
        self.L              = 0.0065             # Temperature lapse rate in K/m
        self.P_sea_level    = 100                # Power at sea level (kW)
        self.mu             = self.Vf*(self.omega*self.MRR)

    def rho_finder(self, h):
        # Calculate air density at altitude
        T = self.T0 - self.L * h  # Temperature at altitude
        rho = self.rho_0 * ((T / self.T0) ** ((self.g / (self.L * self.R0)) - 1))  # Density at altitude
        return rho

    def Forward_Flight_Performance(self,rho, Vf):
        rho = self.rho_finder(self.Altitude)
        power_available = self.P_sea_level * (rho / self.rho_0)
        P_prof = (self.MRA * rho * self.solidity * self.C_d * ((self.omega * self.MRR) ** 3) * (1 + 3 * (self.mu ** 2))) / 8  # in Watts
        P_induced = (self.VW ** 2) / (2 * rho * self.MRA * self.Vf)     # in Watts
        f = 0.37 / math.pi                                              # Flat plate area
        P_parasite = 0.5 * rho * f * (self.Vf ** 3)                     # W
        P_R = P_induced + P_prof + P_parasite                           # W

        # Calculate Rate of Climb (RC)
        rc = ((power_available * 1000 * (1 - self.installed_power_loss)) - P_R) / self.GW  # in m/s

        # Maximum speed based on blade stall
        Vmax = ((2 * self.VW) / (rho * self.C_L * self.MRA)) ** 0.5  # in m/s

        # Calculate Range (km)
        Range = (self.FW * Vf) / (P_R * self.SFC)  # in km
        Flight_time = Range * 1000 / (3600 * self.Vf)  # Maximum flight time in hours

        # Check if fuel is sufficient for mission
        if self.distance > Range:
            raise ValueError("Mission failed: Insufficient fuel")
        
        return power_available, P_induced, P_prof, P_parasite, P_R, Range, Flight_time, rc, Vmax

    def Power_vs_Vf(self, rho, Vf):
        Vf_range = np.linspace(15, 100, 1000)  # Forward velocity variation in m/s

        P_i     = []
        P_pr    = []
        P_para  = []
        P_req   = []
        P_avail = []
        r_c     = []

        for Vf in Vf_range:
            self.Vf = Vf  # Update forward velocity
            rho = self.rho_finder(self.Altitude)
            results = self.Forward_Flight_Performance(rho)
            P_induced, P_prof, P_parasite, P_R, power_available, range, Flight_time, rc, Vmax = results

            P_i.append(P_induced / 1000)
            P_pr.append(P_prof / 1000)
            P_para.append(P_parasite / 1000)
            P_req.append(P_R / 1000)
            P_avail.append(power_available)
            r_c.append(rc)

            if -0.01 <= (power_available - P_R / 1000) <= 0.01:
                print(f"Maximum speed based on power required (m/s): {Vf}")

        plt.plot(Vf_range, P_i, label='Induced Power (kW)', color='blue')
        plt.plot(Vf_range, P_pr, label='Profile Power (kW)', color='orange')
        plt.plot(Vf_range, P_para, label='Parasite Power (kW)', color='green')
        plt.plot(Vf_range, P_req, label='Power Required (kW)', color='red')
        plt.plot(Vf_range, P_avail, label='Power Available (kW)', color='black')
        plt.title('Power vs Forward Speed')
        plt.xlabel('Forward Speed (m/s)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid()
        return plt.show()

# Example usage:
# # Create a simulation and mission input objects
# simulator_inputs = {
#     'GW': 300 * 9.81,  # Take-off weight in Newtons
#     'fuel_weight': 50,  # Fuel weight in kg
#     'h': 2000,  # Altitude in meters
#     'Vf': 30,  # Forward velocity in m/s
#     'distance': 0,  # Distance in km
#     'D': 0,  # Drag in Newtons
#     'installed_power_loss': 0  # Installed power loss
# }
# mission_inputs = {
#     'SFC': 0.36 / 1000,  # Specific fuel consumption in kg/(W*h)
# }
# blade = {
#     'b': 3,  # Number of blades
#     'c': 0.5,  # Chord length in meters
#     'C_d': 0.0079,  # Drag coefficient
#     'C_L': 0.1  # Lift coefficient
# }

# # Initialize and run the simulation
# flight_perf = Forward_Flight_Performance(simulator_inputs, mission_inputs, atmosphere=None, blade=blade)
# flight_perf.Power_vs_Vf()


# class Mission_Segments():
#     def __init__(self) -> None:
#         pass