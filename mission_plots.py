import json
import numpy as np
import matplotlib.pyplot as plt
from Mission_Planner import MissionCoordinator
from U_inputs import U_Inputs_Simulator, Pilot_Inputs

# Load input files
with open("Helicopter_params.json", "r") as f:
    params = json.load(f)

# Initialize simulator inputs and pilot inputs
simulator_inputs = U_Inputs_Simulator(**params["simulator_inputs"])
pilot_inputs = Pilot_Inputs(**params["pilot_inputs"])

# Initialize MissionCoordinator and execute the mission
mission_coordinator = MissionCoordinator(simulator_inputs, pilot_inputs)

# Load mission trajectory
with open("mission_trajectory.json", "r") as f:
    trajectory = json.load(f)

# Run the mission and get the log
mission_log = mission_coordinator.run_mission(trajectory["mission"])

# Variables to plot
variables = {
    "altitudes": ("Altitude", "Altitude (m)"),
    "fuel_weights": ("Fuel Weight", "Fuel weight (kg)"),
    "power_used": ("Power Usage", "Power (kW)"),
    "distances": ("Distance Covered", "Distance (m)"),
    "times": ("Time", "Time (s)"),
    "gross_weight": ("Gross Weight", "Gross Weight (kg)"),
    "fuel_burn_rate": ("Fuel Burn Rate", "Burn Rate (kg/s)"),
    "climb_rate": ("Climb Rate", "Climb Rate (m/s)"),
    "speed": ("Speed", "Speed (m/s)")
}

# Create a 3x3 grid for 9 plots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Mission Logs")

# Flatten axs for easier iteration
axs = axs.flatten()

# Plot each variable
for i, (key, (title, ylabel)) in enumerate(variables.items()):
    if key in mission_log:
        axs[i].plot(mission_log["times"], mission_log[key], label=title)
        axs[i].set_title(title)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlabel("Time (s)")
        axs[i].grid(True)

# Hide any unused subplots
for j in range(len(variables), len(axs)):
    axs[j].axis("off")

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mission_plots.png")
plt.show()
