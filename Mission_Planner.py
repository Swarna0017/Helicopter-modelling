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
        print(f"Endurance (hours): {endurance:.3f}")
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

class Forward_Flight():
    def __init__(self) -> None:
        pass

class Mission_Segments():
    def __init__(self) -> None:
        pass