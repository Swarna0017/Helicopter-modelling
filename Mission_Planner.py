from Blade_G    import Blade
from AirData    import Atmosphere
from U_inputs   import *
from Inflow     import v_calculator
from Airfoil import Airfoil_data
from Instantaneous_Integrator import BEMT_Implementer, Forward_flight_analyzer

import math
import numpy as np
import matplotlib.pyplot as plt


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

class Forward_Flight():
    def __init__(self, simulator_inputs: U_Inputs_Simulator, mission_inputs: U_Inputs_Planner, atmosphere: Atmosphere, blade: Blade):
        # Extract simulator and mission parameters
        self.VW             = simulator_inputs.VW
        self.FW             = simulator_inputs.FW
        self.MR_chord       = simulator_inputs.MR_chord
        self.Altitude       = simulator_inputs.Altitude
        self.SFC            = mission_inputs.SFC
        self.MRR            = simulator_inputs.MRR
        self.MRA            = simulator_inputs.MRA
        self.rpm            = (simulator_inputs.MR_omega)*30/math.pi
        self.solidity       = blade.Disk_solidity(self)
        self.omega          = simulator_inputs.MR_omega
        self.C_d            = simulator_inputs.Blade_Cd
        self.C_L            = simulator_inputs.Blade_Cl
        self.Vf             = simulator_inputs.Vf
        self.distance       = 0                # Expected distance, placeholder value
        self.D              = 0                 # Drag
        self.power_loss     = mission_inputs.Power_loss
        self.rho_0          = 1.225              # Sea level air density in kg/m^3
        self.g              = 9.81               # Gravitational acceleration in m/s^2
        self.R0             = 287.05             # Specific gas constant for air in J/(kg*K)
        self.T0             = 288.15             # Sea level standard temperature in K
        self.L              = 0.0065             # Temperature lapse rate in K/m
        self.P_sea_level    = 100                # Power at sea level (kW)
        self.mu             = self.Vf/(self.omega*self.MRR)

    def rho_finder(self, h):
        # Calculate air density at altitude
        T = self.T0 - self.L * h  # Temperature at altitude
        rho = self.rho_0 * ((T / self.T0) ** ((self.g / (self.L * self.R0)) - 1))  # Density at altitude
        return rho

    def Forward_Flight_Performance(self, rho, Vf):
        rho = self.rho_finder(self.Altitude)
        power_available = self.P_sea_level * (rho / self.rho_0)
        P_prof = (self.MRA * rho * self.solidity * self.C_d * ((self.omega * self.MRR) ** 3) * (1 + 3 * (self.mu ** 2))) / 8  # in Watts
        P_induced = (self.VW ** 2) / (2 * rho * self.MRA * Vf)     # in Watts
        f = 0.37 / math.pi                                              # Flat plate area
        P_parasite = 0.5 * rho * f * (Vf ** 3)                     # W
        P_R = P_induced + P_prof + P_parasite                           # W

        # Calculate Rate of Climb (RC)
        rc = ((power_available * 1000 * (1 - self.power_loss)) - P_R) / self.VW  # in m/s

        # Maximum speed based on blade stall
        Vmax = ((2 * self.VW) / (rho * self.C_L * self.MRA)) ** 0.5  # in m/s

        # Calculate Range (km)
        Range = (self.FW * Vf) / (P_R * self.SFC)  # in km
        Flight_time = Range * 1000 / (3600 * self.Vf)  # Maximum flight time in hours

        # Check if fuel is sufficient for mission
        if self.distance > Range:
            raise ValueError("Mission failed: Insufficient fuel")
        
        return power_available, P_induced, P_prof, P_parasite, P_R, Range, Flight_time, rc, Vmax

    def Power_vs_Vf(self):
        Vf_range = np.linspace(15, 100, 10)  # Forward velocity variation in m/s

        P_i     = []
        P_pr    = []
        P_para  = []
        P_req   = []
        P_avail = []
        r_c     = []

        for Vf in Vf_range:
            rho = self.rho_finder(self.Altitude)
            results = self.Forward_Flight_Performance(rho, Vf)
            power_available, P_induced, P_prof, P_parasite, P_R, range, Flight_time, rc, Vmax = results

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

class MissionCoordinator:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, pilot_inputs: Pilot_Inputs):
        self.simulator_inputs = simulator_inputs
        self.pilot_inputs = pilot_inputs

        # Fuel energy content in J/kg and initialize with provided fuel weight
        self.fuel_energy_content = 43.124 * 1e6  # J/kg
        self.fuel_weight = simulator_inputs.FW
        self.payload_weight = simulator_inputs.VW
        self.gross_weight = self.payload_weight + self.fuel_weight
        self.altitude = simulator_inputs.Altitude

        # Environment setup
        self.atmosphere = Atmosphere(simulator_inputs, pilot_inputs)

        # Blade and aerodynamic components
        self.blade = Blade(simulator_inputs, pilot_inputs)
        self.airfoil = Airfoil_data(simulator_inputs, self.atmosphere)

        # Propeller efficiency estimations
        self.hover_solver = BEMT_Implementer(simulator_inputs, pilot_inputs, self.blade, self.atmosphere, self.airfoil)

        # Mission logs for output analysis
        self.mission_log = {
            "altitudes": [],
            "fuel_weights": [],
            "power_used": [],
            "distances": [],
            "times": [],
        }

    def calculate_available_power(self):
        # Calculate available power at the current altitude
        density_ratio = self.atmosphere.rho_calc() / 1.225  # Standard sea-level density
        max_power = (density_ratio * self.simulator_inputs.MRA * 1e3)  # Scale with density
        return max_power

    def calculate_fuel_consumption(self, power, duration):
        # Calculate fuel consumption based on power (kW) and duration (seconds)
        sfc = self.simulator_inputs.SFC  # kg/kWh
        fuel_used = sfc * power * duration / 3600  # Convert time to hours
        return fuel_used

    def initialize_simulation(self):
        # Log the starting state of the mission
        self.mission_log["altitudes"].append(self.altitude)
        self.mission_log["fuel_weights"].append(self.fuel_weight)
        self.mission_log["power_used"].append(0)
        self.mission_log["distances"].append(0)
        self.mission_log["times"].append(0)

    def update_mission(self, power_used, distance_covered, time_step):
        # Update mission log and internal state
        fuel_used = self.calculate_fuel_consumption(power_used, time_step)
        self.fuel_weight -= fuel_used
        self.gross_weight -= fuel_used

        # Check for limits
        if self.fuel_weight < 0:
            raise Exception("Fuel exceeded! Mission cannot continue.")
        if power_used > self.calculate_available_power():
            raise Exception("Power exceeded! Mission cannot continue.")

        # Log updates
        self.mission_log["fuel_weights"].append(self.fuel_weight)
        self.mission_log["power_used"].append(power_used)
        self.mission_log["distances"].append(self.mission_log["distances"][-1] + distance_covered)
        self.mission_log["times"].append(self.mission_log["times"][-1] + time_step)

    def simulate_hover(self, duration):
        # Simulate hover for the given duration
        thrust = self.gross_weight * 9.81  # N
        power_required = self.hover_solver.Power_Calculator()
        self.update_mission(power_required, 0, duration)

    def simulate_forward_flight(self, velocity, distance):
        # Simulate forward flight for a given distance and velocity
        flight_time = distance / velocity
        power_solver = Forward_flight_analyzer(self.simulator_inputs, self.pilot_inputs, self.blade, self.atmosphere)
        power_required = power_solver.Power_Calculator()

        for t in np.arange(0, flight_time, 1):
            self.update_mission(power_required, velocity, 1)

    def handle_takeoff(self, hover_duration):
        # Handle initial hover phase during takeoff
        self.simulate_hover(hover_duration)

    def handle_climb(self, climb_rate, target_altitude):
        # Simulate vertical climb
        climb_time = (target_altitude - self.altitude) / climb_rate
        thrust = self.gross_weight * 9.81
        power_solver = BEMT_Implementer(self.simulator_inputs, self.pilot_inputs, self.blade, self.atmosphere)
        power_required = power_solver.Power_Calculator()

        for t in np.arange(0, climb_time, 1):
            self.update_mission(power_required, 0, 1)
        self.altitude = target_altitude

    def run_mission(self, mission_plan):
        self.initialize_simulation()
        for phase in mission_plan:
            if phase["type"] == "takeoff":
                self.handle_takeoff(phase["duration"])
            elif phase["type"] == "climb":
                self.handle_climb(phase["climb_rate"], phase["target_altitude"])
            elif phase["type"] == "forward_flight":
                self.simulate_forward_flight(phase["velocity"], phase["distance"])

        # Return mission log for analysis
        return self.mission_log