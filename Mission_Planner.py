# mission_planner.py
import math
import numpy as np
import matplotlib.pyplot as plt
from Blade_G import Blade
from AirData import Atmosphere
from U_inputs import U_Inputs_Simulator, U_Inputs_Planner
from Inflow import v_calculator

P_sea_level = 250  # Available power MSL, kW

class Hover_Climb():
    def __init__(self, simulator_inputs: U_Inputs_Simulator, mission_inputs: U_Inputs_Planner, atmosphere: Atmosphere, blade: Blade):
        self.VW=mission_inputs.VW
        self.FW=mission_inputs.FW
        self.Altitude=simulator_inputs.Altitude
        self.rho=atmosphere.rho_calc()
        self.T1=atmosphere.T1_calc()
        self.rho_0=1.225 # MSL, kg/m^3
        self.solidity=blade.Disk_solidity()
        self.omega=simulator_inputs.MR_omega
        self.Cd=0.3
        self.MRA=simulator_inputs.MRA
        self.SFC=mission_inputs.SFC
        self.hover_time=0.0001

    def Performance(self):
        P_available = P_sea_level * (self.rho / self.rho_0)
        P_prof = (self.rho * self.solidity * self.Cd * ((self.omega * self.MRR) ** 3) * math.pi * (self.MRR ** 2)) / 8  # W
        P_induced = ((self.VW ** 3) / (2 * self.rho * self.MRA)) ** 0.5  # W
        P_R = P_induced + P_prof  # in Watts

        # Calculate Rate of Climb (rc)
        rc = (P_available * 1000 - P_R) / self.VW  # in m/s

        # Calculate endurance (hours)
        fuel_flow_rate = self.SFC * P_R  # in kg/s
        endurance = self.FW / (3600 * fuel_flow_rate)  # in hours

        # Check if fuel available is sufficient for the mission
        if self.hover_time > endurance:
            return "Mission failed: Insufficient fuel"
        else:
            return P_available, P_induced, P_prof, P_R, endurance, rc


    def Power_Outputs(simulator_inputs, mission_inputs, atmosphere, blade):
        # Create an instance of Hover_Climb
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)

        # Call the Hover_and_Climb_Performance function
        P_available, P_induced, P_prof, P_R, endurance, rc = hover_climb.Performance()

        # Output mission results
        print(f"Induced Power (kW): {P_induced / 1000:.3f}")
        print(f"Profile Power (kW): {P_prof / 1000:.3f}")
        print(f"Total Power required (kW): {P_R / 1000:.3f}")
        print(f"Power Available (kW): {P_available:.3f}")
        print(f"Endurance (hours): {endurance:.3f}")
        print(f"Maximum Rate of Climb (m/s): {rc:.3f}")

        return P_available, P_induced, P_prof, P_R, endurance, rc

    # Mission planner for multiple altitudes (plotting)
    def Power_vs_Alt(self, simulator_inputs, mission_inputs, atmosphere, blade):
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        altitudes = np.linspace(0, 20000, 20000)  # Altitude variation in meters

        # Lists to store values for plotting
        P_i = []
        P_pr = []
        P_req = []
        P_avail = []
        r_c = []

        for h in altitudes:
            P_available, P_induced, P_prof, P_R, endurance, rc = hover_climb.Performance(self)
            P_i.append(P_induced / 1000)
            P_pr.append(P_prof / 1000)
            P_req.append(P_R / 1000)
            P_avail.append(P_available)
            r_c.append(rc)

            # Output ceiling information
            if -0.001 <= rc <= 0.001:
                print(f"Absolute ceiling (m): {h:.3f}")
            elif 0.507 <= rc <= 0.509:
                print(f"Service ceiling (m): {h:.3f}")

        # Plot all kinds of power vs altitude
        plt.plot(altitudes, P_i, label='Induced Power (kW)', color='blue')
        plt.plot(altitudes, P_pr, label='Profile Power (kW)', color='orange')
        plt.plot(altitudes, P_req, label='Power Required (kW)', color='red')
        plt.plot(altitudes, P_avail, label='Power Available (kW)', color='black')
        plt.title('Power vs Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid()
        plt.xlim(0, 8000)
        return plt.show()

    def RC_vs_Alt(self, simulator_inputs, mission_inputs, atmosphere, blade):
        r_c = []
        P_available, P_induced, P_prof, P_R, endurance, rc = hover_climb.Performance(self)
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        altitudes = np.linspace(0, 20000, 20000)
        for h in altitudes:
            _, _, _, _, _, rc = hover_climb.Performance(self)
            r_c.append(rc)
        # Plot Rate of Climb vs Altitude
        plt.plot(altitudes, r_c, label='Rate of Climb (m/s)', color='blue')
        plt.title('R/C vs Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('Rate of Climb (m/s)')
        plt.legend()
        plt.grid()
        plt.xlim(0, 8000)
        plt.ylim(-10, 25)
        return plt.show()


    # Mission planner for multiple take-off weights (plotting)
    def RC_vs_weight(self, simulator_inputs, mission_inputs, atmosphere, blade):
        hover_climb = Hover_Climb(simulator_inputs, mission_inputs, atmosphere, blade)
        # List to store climb rates
        r_c = []
        VW = np.linspace(1, 400 * 9.81, 1000)  # Take-off weight variation in Newtons

        for W in VW:
            P_available, P_induced, P_prof, P_R, endurance, rc = hover_climb.Performance(self)
            r_c.append(rc)

        # Plot Rate of Climb vs Take-off weight
        plt.plot(VW, r_c, label='Rate of Climb (m/s)', color='blue')
        plt.title('Rate of Climb vs Gross weight')
        plt.xlabel('Gross weight (N)')
        plt.ylabel('Rate of Climb (m/s)')
        plt.legend()
        plt.grid()
        plt.xlim(1, 400 * 9.81)
        plt.show()
