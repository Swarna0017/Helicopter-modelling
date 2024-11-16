from Blade_G    import Blade
from AirData    import Atmosphere
from U_inputs   import U_Inputs_Simulator, U_Inputs_Planner
from Inflow     import v_calculator

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

# This section simulates/co-ordinates the missions given in assignment 3 
GRAVITY = 9.81  # Gravity acceleration (m/s^2)
DESIRED_CLIMB_RATE = 2.0  # m/s, desired climb rate for successful missions
DESIRED_CRUISE_SPEED = 60.0  # m/s, desired cruise speed for level flight
FUEL_CONSUMPTION_RATE = 0.1  # Fuel consumption rate (kg/s)

# Helicopter performance constants (example values)
MAX_ENGINE_POWER = 500  # kW, example maximum power available
MAX_FUEL_CAPACITY = 1000  # kg, maximum fuel capacity

# Mission scenarios (as per your previous context)
missions = [
    {"name": "Successful Payload Drop", "task": "vertical climb, steady climb"},
    {"name": "Successful Payload Pickup", "task": "steady climb, vertical climb"},
    {"name": "Fuel-Limited Unsuccessful Payload Pickup", "task": "steady climb, vertical climb, fuel constraints"},
    {"name": "Power-Limited Unsuccessful Payload Drop", "task": "level flight, power constraints"}
]

# Auxiliary functions for fuel and weight calculations
def calculate_air_density(temperature, pressure):
    """Simple function to calculate air density using the ideal gas law (simplified)."""
    R = 287.05  # Specific gas constant for dry air (J/(kg·K))
    temp_kelvin = temperature + 273.15
    air_density = pressure / (R * temp_kelvin)
    return air_density

def vertical_climb(initial_weight, fuel_weight, climb_rate):
    """Calculate the vertical climb phase."""
    max_altitude = 1000  # m
    time_to_max_altitude = max_altitude / climb_rate
    gross_weight_at_max_altitude = initial_weight - fuel_weight
    return time_to_max_altitude, gross_weight_at_max_altitude

def steady_climb(initial_weight, fuel_weight, climb_rate, wind_speed):
    """Calculate the steady climb phase with wind conditions."""
    net_climb_rate = climb_rate - wind_speed
    max_altitude = 1000  # m
    time_to_max_altitude = max_altitude / net_climb_rate
    gross_weight_at_max_altitude = initial_weight - fuel_weight
    return time_to_max_altitude, gross_weight_at_max_altitude

def level_flight(initial_weight, fuel_weight, cruise_speed):
    """Calculate the level flight phase."""
    time_to_runout = fuel_weight / FUEL_CONSUMPTION_RATE
    distance_covered = cruise_speed * time_to_runout
    fuel_consumed = FUEL_CONSUMPTION_RATE * time_to_runout
    return distance_covered, fuel_consumed

def power_limited_climb(initial_weight, fuel_weight, climb_rate, available_power):
    """Check if climb rate is achievable based on available power."""
    power_required_for_climb = (initial_weight * GRAVITY * climb_rate) / available_power
    if power_required_for_climb > available_power:
        return False  # Not enough power for the climb
    return True


def plot_mission_results(mission_results, mission_type, initial_weight, fuel_burn_rate):
    """Plot mission results (e.g., weight, fuel consumption, etc.) in multiple subplots."""
    
    # Initialize lists to store times, weights, and other metrics for plotting
    times = []
    weights = []
    fuel_consumed = []
    distances = []
    climb_rates = []

    current_weight = initial_weight  # Start with initial gross weight
    total_fuel_consumed = 0  # Initialize fuel consumed

    # Process mission results
    for result in mission_results:
        if len(result) == 2:  # Vertical and Steady Climb (time, weight)
            time, weight = result
            times.append(time)
            weights.append(weight)
            fuel_consumed.append(total_fuel_consumed)
            distances.append(0)  # No distance covered in climb phase
            climb_rates.append(weight)  # Use weight as a placeholder for climb rate (can refine later)
        elif len(result) == 3:  # Level Flight (distance, fuel consumption)
            distance, fuel = result
            # Calculate time from distance (distance / speed)
            time = distance / DESIRED_CRUISE_SPEED
            total_fuel_consumed += fuel
            current_weight = initial_weight - total_fuel_consumed
            times.append(time)
            weights.append(current_weight)
            fuel_consumed.append(total_fuel_consumed)
            distances.append(distance)
            climb_rates.append(0)  # No climb during level flight, so no climb rate

    # Ensure times and weights have the same dimensions for plotting
    if len(times) != len(weights):
        print(f"Error: Times and Weights have different lengths for {mission_type}.")
        return

    # Create subplots in a 2x2 grid
    plt.figure(figsize=(12, 10))

    # Plot 1: Gross Weight vs Time
    plt.subplot(2, 2, 1)
    plt.plot(times, weights, label=f'{mission_type} Gross Weight', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Weight (kg)')
    plt.title(f'{mission_type} - Gross Weight Over Time')
    plt.grid(True)
    plt.legend()

    # Plot 2: Fuel Consumption vs Time
    plt.subplot(2, 2, 2)
    plt.plot(times, fuel_consumed, label=f'{mission_type} Fuel Consumption', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Fuel Consumed (kg)')
    plt.title(f'{mission_type} - Fuel Consumption Over Time')
    plt.grid(True)
    plt.legend()

    # Plot 3: Climb Rate vs Time (placeholder for vertical climb or steady climb rates)
    plt.subplot(2, 2, 3)
    plt.plot(times, climb_rates, label=f'{mission_type} Climb Rate', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Climb Rate (m/s)')
    plt.title(f'{mission_type} - Climb Rate Over Time')
    plt.grid(True)
    plt.legend()

    # Plot 4: Distance Covered vs Time (level flight)
    plt.subplot(2, 2, 4)
    plt.plot(times, distances, label=f'{mission_type} Distance Covered', color='c')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title(f'{mission_type} - Distance Covered Over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # Adjust layout for neat display
    plt.show()


# Mission Execution and Planning
def execute_mission(mission):
    """Execute the mission based on its task requirements."""
    print(f"Executing mission: {mission['name']}")

    initial_weight = 50  # kg, example initial weight
    fuel_weight = 10     # kg, example initial fuel weight
    mission_results = []

    # Check for fuel constraints
    if "fuel constraints" in mission['task']:
        if fuel_weight < 100:
            print("Mission fuel-limited: Insufficient fuel for climb.")
            return mission_results

    # Check for power constraints
    if "power constraints" in mission['task']:
        if not power_limited_climb(initial_weight, fuel_weight, DESIRED_CLIMB_RATE, MAX_ENGINE_POWER):
            print("Mission power-limited: Not enough power for climb.")
            return mission_results

    # Execute the segments based on the task requirements
    if "vertical climb" in mission['task']:
        time_to_max_altitude, gross_weight_at_max = vertical_climb(initial_weight, fuel_weight, DESIRED_CLIMB_RATE)
        mission_results.append((time_to_max_altitude, gross_weight_at_max))
        print(f"Time to reach max altitude (vertical climb): {time_to_max_altitude} seconds")
        print(f"Gross weight at max altitude: {gross_weight_at_max} kg")

    if "steady climb" in mission['task']:
        wind_speed = 5.0  # m/s, example wind speed
        time_to_max_altitude, gross_weight_at_max = steady_climb(initial_weight, fuel_weight, DESIRED_CLIMB_RATE, wind_speed)
        mission_results.append((time_to_max_altitude, gross_weight_at_max))
        print(f"Time to reach max altitude (steady climb): {time_to_max_altitude} seconds")
        print(f"Gross weight at max altitude: {gross_weight_at_max} kg")

    if "level flight" in mission['task']:
        distance, fuel_used = level_flight(initial_weight, fuel_weight, DESIRED_CRUISE_SPEED)
        mission_results.append((distance, fuel_used))
        print(f"Distance covered during level flight: {distance} meters")
        print(f"Fuel consumed during level flight: {fuel_used} kg")

    plot_mission_results(mission_results, mission['name'], initial_weight=50, fuel_burn_rate=0.25)
    return mission_results

# Main execution loop for all missions
if __name__ == "__main__":
    for mission in missions:
        execute_mission(mission)
