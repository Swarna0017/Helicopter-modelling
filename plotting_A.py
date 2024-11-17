import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from Heli_dyn import HelicopterModel, MissionCoordinator

# Load helicopter and mission configurations from JSON files
with open('helicopter_params.json', 'r') as f:
    inputs = json.load(f)
helicopter = HelicopterModel(inputs)

mission = MissionCoordinator()
with open('mission_A.json', 'r') as f:
    mission_params = json.load(f)

# Define conditions
conditions = [
    {
        'weight': 400,
        'V': 10,
        'alt': 2000,
        'theta_0': np.radians(8.0),
        'theta_1c': np.radians(4.0),
        'theta_1s': np.radians(6.0),
        'theta_tail': np.radians(3.0)
    }
]

# Assuming mission.evaluate_mission_parameters returns a dictionary with required data
plots = mission.evaluate_mission_parameters(mission_params, inputs, conditions)

# Define the variables to plot and their corresponding labels
variables = {
    'altitudes': ('Altitude', 'Altitude (m)'),
    'fuel_weight': ('Fuel Weight', 'Fuel weight (kg)'),
    'gross_weight': ('Gross Weight', 'Gross weight (kg)'),
    'fuel_burn_rate': ('Fuel Burn Rate', 'Fuel Burn Rate (kg/s)'),
    'climb_rate': ('Climb Rate', 'Climb Rate (m/s)'),
    'powers': ('Power Consumption', 'Power (KW)'),
    'available_engine_power': ('Available Engine Power', 'Available Power (KW)'),
    'speed': ('Speed', 'Speed (m/s)'),
    'distance_covered': ('Distance Covered', 'Distance Covered (m)')
}

# Define the figure and number of subplots (3 rows and 3 columns)
fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=False)
fig.suptitle('Mission Plots wrt Time')

# Flatten axs array for easier indexing
axs = axs.flatten()

# Limit the number of variables to the available subplots (9 in total)
max_plots = len(axs)
variables_to_plot = list(variables.items())[:max_plots]

time_interval = 5  # Adjust time interval if needed

for i, (key, (title, ylabel)) in enumerate(variables_to_plot):
    # Check if the variable exists in the plots dictionary
    if key in plots:
        if key == 'gross_weight':
            axs[i].plot(plots['time'][1:], plots[key], 'b')
        elif key == 'altitudes':
            axs[i].plot(np.arange(len(plots[key])) * time_interval, plots[key], 'b')

        else:
            axs[i].plot(np.arange(len(plots[key])) * time_interval, plots[key], 'b')
        
        axs[i].set_title(title)
        axs[i].set_ylabel(ylabel)
        axs[i].grid(True)
        axs[i].set_xlabel('Time (s)')  # Separate x-axis label for each plot

# Hide any unused subplots (if any)
for j in range(len(variables_to_plot), len(axs)):
    axs[j].axis('off')

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the main title
plt.savefig("Mission_A_Altitude_Exceeded.png")
plt.show()
