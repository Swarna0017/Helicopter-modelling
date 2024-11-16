import numpy as np
import matplotlib.pyplot as plt

# Atmosphere Class
class Atmosphere:
    def __init__(self):
        self.g = 9.81  # Gravity [m/s^2]
        self.rho0 = 1.225  # Sea level air density [kg/m^3]
        self.temp0 = 288.15  # Sea level temperature [K]
        self.lapse_rate = -0.0065  # Temperature lapse rate [K/m]
        self.R = 287.05  # Gas constant [J/(kg·K)]

    def density(self, altitude):
        # Calculate air density based on altitude using standard atmosphere equations
        temp = self.temp0 + self.lapse_rate * altitude
        pressure = (101325 * (temp / self.temp0) ** (-self.g / (self.R * self.lapse_rate)))
        rho = pressure / (self.R * temp)
        return rho

    def wind_effect(self, wind_speed, direction):
        # Returns wind adjustment (positive for tailwind, negative for headwind)
        return wind_speed if direction == "tailwind" else -wind_speed


# Helicopter Model Class
class Helicopter:
    def __init__(self, max_takeoff_weight, fuel_burn_rate, engine_power):
        self.max_takeoff_weight = max_takeoff_weight  # Max takeoff weight [kg]
        self.fuel_burn_rate = fuel_burn_rate  # Fuel consumption rate [kg/hour]
        self.engine_power = engine_power  # Total available engine power [W]

    def power_required(self, weight, air_density, airspeed):
        # Simplified power calculation (hover, climb, or cruise)
        drag = 0.5 * air_density * airspeed**2 * 0.3  # Assumed drag coefficient
        induced_power = weight**2 / (2 * air_density * np.pi * 10**2)  # Rotor-induced power
        total_power = drag * airspeed + induced_power
        return min(total_power, self.engine_power)  # Cap at engine power

    def fuel_consumption(self, power, time):
        # Fuel consumption based on engine power usage
        return (power / self.engine_power) * self.fuel_burn_rate * time / 60  # Convert to kg/minute


# Mission Coordinator Class
class MissionCoordinator:
    def __init__(self, helicopter, atmosphere):
        self.helicopter = helicopter
        self.atmosphere = atmosphere
        self.time_log = []
        self.altitude_log = []
        self.fuel_log = []
        self.gross_weight_log = []
        self.burn_rate_log = []

    def execute_phase(self, phase_name, duration, altitude, airspeed, payload=0, wind_speed=0, wind_direction="none"):
        # Logs data for each mission phase
        wind_adjustment = self.atmosphere.wind_effect(wind_speed, wind_direction)
        air_density = self.atmosphere.density(altitude)
        adjusted_airspeed = airspeed + wind_adjustment

        # Calculate phase-specific power and fuel consumption
        power = self.helicopter.power_required(self.helicopter.max_takeoff_weight, air_density, adjusted_airspeed)
        fuel_consumed = self.helicopter.fuel_consumption(power, duration)
        self.helicopter.max_takeoff_weight -= fuel_consumed  # Update weight

        # Log data
        self.time_log.append(duration)
        self.altitude_log.append(altitude)
        self.fuel_log.append(fuel_consumed)
        self.gross_weight_log.append(self.helicopter.max_takeoff_weight)
        self.burn_rate_log.append(fuel_consumed / duration)

    def run_mission(self, mission_phases):
        # Iterate through mission phases
        for phase in mission_phases:
            self.execute_phase(**phase)

    def plot_results(self):
        # Create a 2x4 grid of plots
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        axs = axs.flatten()

        # Plot each mission parameter
        axs[0].plot(self.time_log, self.gross_weight_log, label="Gross Weight (kg)")
        axs[0].set_title("Gross Weight vs Time")
        axs[0].set_xlabel("Time (min)")
        axs[0].set_ylabel("Weight (kg)")

        axs[1].plot(self.time_log, self.fuel_log, label="Fuel Weight (kg)")
        axs[1].set_title("Fuel Weight vs Time")
        axs[1].set_xlabel("Time (min)")
        axs[1].set_ylabel("Fuel (kg)")

        axs[2].plot(self.time_log, self.burn_rate_log, label="Fuel Burn Rate")
        axs[2].set_title("Fuel Burn Rate vs Time")
        axs[2].set_xlabel("Time (min)")
        axs[2].set_ylabel("Burn Rate (kg/min)")

        axs[3].plot(self.time_log, self.altitude_log, label="Altitude (AMSL)")
        axs[3].set_title("Altitude vs Time")
        axs[3].set_xlabel("Time (min)")
        axs[3].set_ylabel("Altitude (m)")

        # Adjust grid and show
        for ax in axs:
            ax.legend()
            ax.grid()
        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Initialize atmosphere and helicopter
    atmosphere = Atmosphere()
    helicopter = Helicopter(max_takeoff_weight=3000, fuel_burn_rate=120, engine_power=100000)

    # Define mission phases
    mission_phases = [
        {"phase_name": "Takeoff", "duration": 5, "altitude": 2000, "airspeed": 0, "payload": 50},
        {"phase_name": "Climb", "duration": 10, "altitude": 2500, "airspeed": 10, "wind_speed": 20, "wind_direction": "tailwind"},
        {"phase_name": "Cruise", "duration": 20, "altitude": 2500, "airspeed": 50},
        {"phase_name": "Descent", "duration": 10, "altitude": 2300, "airspeed": 20, "wind_speed": 20, "wind_direction": "headwind"},
        {"phase_name": "Hover", "duration": 1, "altitude": 2300, "airspeed": 0},
    ]

    # Run mission and plot results
    coordinator = MissionCoordinator(helicopter, atmosphere)
    coordinator.run_mission(mission_phases)
    coordinator.plot_results()
