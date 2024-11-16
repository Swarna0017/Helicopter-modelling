import matplotlib.pyplot as plt

# Constants and parameters
GRAVITY = 9.81  # Gravity acceleration (m/s^2)
DESIRED_CLIMB_RATE = 20.0  # m/s, desired climb rate for successful missions
DESIRED_CRUISE_SPEED = 60.0  # m/s, desired cruise speed for level flight
FUEL_CONSUMPTION_RATE = 0.1  # Fuel consumption rate (kg/s)

# Helicopter performance constants (example values)
MAX_ENGINE_POWER = 200  # kW, example maximum power available
MAX_FUEL_CAPACITY = 100  # kg, maximum fuel capacity

# Mission scenarios
missions = [
    {"name": "Successful Payload Drop", "task": "vertical climb, steady climb"},
    {"name": "Successful Payload Pickup", "task": "steady climb, vertical climb"},
    {"name": "Fuel-Limited Unsuccessful Payload Pickup", "task": "steady climb, vertical climb, fuel constraints"},
    {"name": "Power-Limited Unsuccessful Payload Drop", "task": "level flight, power constraints"}
]

# Auxiliary functions for fuel and weight calculations
def vertical_climb(initial_weight, fuel_weight, climb_rate):
    """Calculate the vertical climb phase."""
    max_altitude = 1000  # m
    time_to_max_altitude = max_altitude / climb_rate
    fuel_used = fuel_weight * (time_to_max_altitude / 1000)  # fuel consumed in this phase
    fuel_used = min(fuel_used, fuel_weight)  # Prevent fuel from exceeding available fuel
    gross_weight_at_max_altitude = initial_weight - fuel_used
    return time_to_max_altitude, gross_weight_at_max_altitude, fuel_used

def steady_climb(initial_weight, fuel_weight, climb_rate, wind_speed):
    """Calculate the steady climb phase with wind conditions."""
    net_climb_rate = climb_rate - wind_speed
    max_altitude = 1000  # m
    time_to_max_altitude = max_altitude / net_climb_rate
    fuel_used = fuel_weight * (time_to_max_altitude / 1000)  # fuel consumed in this phase
    fuel_used = min(fuel_used, fuel_weight)  # Prevent fuel from exceeding available fuel
    gross_weight_at_max_altitude = initial_weight - fuel_used
    return time_to_max_altitude, gross_weight_at_max_altitude, fuel_used

def level_flight(initial_weight, fuel_weight, cruise_speed):
    """Calculate the level flight phase."""
    time_to_runout = fuel_weight / FUEL_CONSUMPTION_RATE
    fuel_consumed = FUEL_CONSUMPTION_RATE * time_to_runout
    fuel_consumed = min(fuel_consumed, fuel_weight)  # Prevent fuel from going negative
    distance_covered = cruise_speed * time_to_runout
    gross_weight_at_runout = initial_weight - fuel_consumed
    return distance_covered, fuel_consumed, gross_weight_at_runout

def power_limited_climb(initial_weight, fuel_weight, climb_rate, available_power):
    """Check if climb rate is achievable based on available power."""
    power_required_for_climb = (initial_weight * GRAVITY * climb_rate) / available_power
    if power_required_for_climb > available_power:
        return False  # Not enough power for the climb
    return True

def plot_mission_results(mission_results, mission_type, initial_weight):
    """Plot mission results (e.g., weight, fuel consumption, etc.) in multiple subplots."""
    
    times, weights, fuel_consumed, distances, climb_rates = [], [], [], [], []
    
    total_fuel_consumed = 0  # Initialize fuel consumed
    current_weight = initial_weight  # Start with initial gross weight

    # Process mission results and update values for each phase
    for result in mission_results:
        if len(result) == 2:  # Vertical and Steady Climb (time, weight)
            time, weight, fuel = result
            times.append(time)
            weights.append(weight)
            fuel_consumed.append(total_fuel_consumed)
            distances.append(0)  # No distance covered in climb phase
            climb_rates.append(weight)  # Use weight as a placeholder for climb rate
        elif len(result) == 3:  # Level Flight (distance, fuel consumption, weight)
            distance, fuel, weight = result
            time = distance / DESIRED_CRUISE_SPEED
            total_fuel_consumed += fuel
            current_weight = initial_weight - total_fuel_consumed
            times.append(time)
            weights.append(current_weight)
            fuel_consumed.append(total_fuel_consumed)
            distances.append(distance)
            climb_rates.append(0)  # No climb during level flight

    # Create subplots in a 2x4 grid for eight plots
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'{mission_type} - Mission Results', fontsize=10)

    # Plot 1: Gross Weight vs Time
    plt.subplot(2, 4, 1)
    plt.plot(times, weights, label='Gross Weight', color='b')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Weight (kg)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 2: Fuel Consumption vs Time
    plt.subplot(2, 4, 2)
    plt.plot(times, fuel_consumed, label='Fuel Consumption', color='g')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Fuel Consumed (kg)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 3: Climb Rate vs Time
    plt.subplot(2, 4, 3)
    plt.plot(times, climb_rates, label='Climb Rate', color='r')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Climb Rate (m/s)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 4: Distance Covered vs Time
    plt.subplot(2, 4, 4)
    plt.plot(times, distances, label='Distance Covered', color='c')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Distance (m)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 5: Fuel Weight vs Time
    plt.subplot(2, 4, 5)
    plt.plot(times, [initial_weight - w for w in weights], label='Fuel Weight', color='m')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Fuel Weight (kg)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 6: Remaining Weight vs Time
    plt.subplot(2, 4, 6)
    plt.plot(times, [initial_weight - fuel for fuel in fuel_consumed], label='Remaining Weight', color='y')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Remaining Weight (kg)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 7: Power Consumption vs Time
    plt.subplot(2, 4, 7)
    power_consumption = [weight * GRAVITY * climb_rate for weight, climb_rate in zip(weights, climb_rates)]
    plt.plot(times, power_consumption, label='Power Consumption', color='orange')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Power Consumption (W)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # Plot 8: Speed vs Time
    plt.subplot(2, 4, 8)
    speed = [DESIRED_CRUISE_SPEED if dist > 0 else 0 for dist in distances]  # Level flight speed or 0 for climb
    plt.plot(times, speed, label='Speed', color='purple')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Speed (m/s)', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    plt.tight_layout()  # Adjust layout for neat display
    plt.show()


def execute_mission(mission):
    """Execute the mission based on its task requirements."""
    print(f"Executing mission: {mission['name']}")

    initial_weight = 500  # kg, example initial weight
    fuel_weight = 100     # kg, example initial fuel weight
    mission_results = []

    # Check for fuel constraints
    if "fuel constraints" in mission['task']:
        if fuel_weight < 100:
            print("Mission fuel-limited: Insufficient fuel for climb.")
            # Provide default mission results with minimal data
            mission_results.append((0, initial_weight, 0))  # No vertical climb
            mission_results.append((0, initial_weight, 0))  # No steady climb
            mission_results.append((0, initial_weight, 0))  # No level flight
            plot_mission_results(mission_results, mission['name'], initial_weight)
            return

    # Check for power constraints
    if "power constraints" in mission['task']:
        if not power_limited_climb(initial_weight, fuel_weight, 10, MAX_ENGINE_POWER):
            print("Mission power-limited: Not enough power for climb.")
            # Provide default mission results with minimal data
            mission_results.append((0, initial_weight, 0))  # No vertical climb
            mission_results.append((0, initial_weight, 0))  # No steady climb
            mission_results.append((0, initial_weight, 0))  # No level flight
            plot_mission_results(mission_results, mission['name'], initial_weight)
            return

    # Execute phases
    vertical_time, vertical_weight, vertical_fuel = vertical_climb(initial_weight, fuel_weight, DESIRED_CLIMB_RATE)
    steady_time, steady_weight, steady_fuel = steady_climb(initial_weight, fuel_weight, DESIRED_CLIMB_RATE, 5)  # Wind speed
    level_distance, level_fuel, level_weight = level_flight(initial_weight, fuel_weight, DESIRED_CRUISE_SPEED)

    mission_results.append((vertical_time, vertical_weight, vertical_fuel))
    mission_results.append((steady_time, steady_weight, steady_fuel))
    mission_results.append((level_distance, level_fuel, level_weight))

    plot_mission_results(mission_results, mission['name'], initial_weight)


# Execute all mission scenarios
for mission in missions:
    execute_mission(mission)
