# Backend for plotting functions
import numpy as np
import math
import pandas as pd
from scipy.optimize import fsolve
import time

class Atmosphere:
    def __init__(self):
        # Sea-level standard conditions initialization
        self.elevation = 0  # in meters
        self.density_sea_level = 1.225  # kg/m^3
        self.temperature_sea_level = 288.15  # Kelvin
        self.temp_lapse_rate = 0.0065  # K/m, temperature lapse rate
        
        # Set initial values based on sea-level conditions
        self.density = self.density_sea_level
        self.temperature = self.temperature_sea_level

    def set_elevation(self, elevation):
        """Set the altitude (elevation) and update atmospheric properties."""
        self.elevation = elevation
        
        self.temperature = self.temperature_sea_level - self.temp_lapse_rate * self.elevation
        
        # Calculate air density at altitude using the barometric formula
        self.density = self.density_sea_level * ((1 - (self.temp_lapse_rate * self.elevation) / self.temperature_sea_level) ** 4.2553)

class Stabilizer:
    def __init__(
        self,inputs
    ):
        
        self.span = inputs['span']
        
        self.c_d0 = inputs['c_d0']
        self.c_la = inputs['c_la']
        self.c_ds = inputs['c_ds']
        
        self.chord = inputs['chord']

        self.x_cg = inputs['x_cg']
        self.z_cg = inputs['z_cg']


    def calculate_chord(self, r):
        m = (self.root_chord - self.tip_chord)/(self.root_cutout - self.radius)
        c = self.root_chord - (m)*self.root_cutout
        return m*r + c
      
    def calculate_lift_coefficient(self, alpha):
        return self.c_la*alpha
    
    def calculate_drag_coefficient(self, alpha):
        return self.c_d0 + self.c_ds*(alpha**2)
    


class Blade:
    def __init__(self, parameters, azimuth):
        # Initialize blade properties
        self.azimuth = self.azimuth_0 = azimuth
        self.radius = parameters['radius']
        self.root_cutout = parameters['root_cutout']
        
        # Aerodynamic characteristics
        self.drag_coefficient_base = parameters['c_d0']
        self.lift_coefficient_slope = parameters['c_l_a']
        self.drag_coefficient_slope = parameters['c_d_b']
        
        # Chord and taper specifications
        self.chord_at_root = parameters['chord']
        self.taper_ratio = parameters['taper_ratio']
        self.chord_at_tip = self.taper_ratio * self.chord_at_root
        
        self.twist_angle_root = np.radians(parameters['twist_angle_root'])
        self.twist_angle_tip = np.radians(parameters['twist_angle_tip'])
        self.twist_difference = self.twist_angle_root - self.twist_angle_tip
            

    def calculate_chord(self, r):
        """Determine the chord length at a specific radial position `r`."""
        slope = (self.chord_at_root - self.chord_at_tip) / (self.root_cutout - self.radius)
        intercept = self.chord_at_root - slope * self.root_cutout
        return slope * r + intercept

    def calculate_twist(self, r):
        """Determine the twist angle at a specific radial position `r`."""
        slope = (self.twist_angle_root - self.twist_angle_tip) / (self.root_cutout - self.radius)
        intercept = self.twist_angle_root - slope * self.root_cutout
        return slope * r + intercept

    def calculate_lift_coefficient(self, alpha):
        """Compute the lift coefficient for a specified angle of attack `alpha`."""
        return self.lift_coefficient_slope * alpha

    def calculate_drag_coefficient(self, alpha):
        """Compute the drag coefficient for a specified angle of attack `alpha`."""
        return self.drag_coefficient_base + self.drag_coefficient_slope * (alpha ** 2)

    
    

class HelicopterModel:

    def __init__(self, config):
        # Initialize the atmosphere and rotor components
        self.atmosphere = Atmosphere()
        self.main_rotor = RotorSystem(config['main'])
        self.horizontal_stabilizer = Stabilizer(config['horizontal_stabilizer'])
        self.vertical_stabilizer = Stabilizer(config['vertical_stabilizer'])
        self.tail_rotor = RotorSystem(config['tail'])

        # Retrieve helicopter specifications
        self.payload_weight = config['helicopter']['payload']
        self.fuel_weight = config['helicopter'].get('fuel', 0)
        self.body_weight = config['helicopter'].get('body_weight', 0)
        self.total_weight = self.payload_weight + self.fuel_weight + self.body_weight
        self.velocity = config['helicopter'].get('V', 0)

        # Gravitational constant
        self.gravity = 9.8

        # Center of gravity positions
        self.main_rotor_cg_x = config['helicopter']['x_mr_cg']
        self.tail_rotor_cg_x = config['helicopter']['x_tr_cg']
        self.main_rotor_cg_z = config['helicopter']['z_mr_cg']
        self.tail_rotor_cg_z = config['helicopter']['z_tr_cg']

        # Flight mode and drag coefficient initialization
        self.flight_mode = 'forward'
        self.fuselage_drag_coefficient = 0.45
        self.moment_of_inertias_first_iteration = True

        self.main_rotor.angular_velocity = 1000 / 9.55
        self.tail_rotor.angular_velocity = 960 / 9.55

    def update_altitude(self, new_altitude):
        # Adjust the altitude in the atmospheric model
        self.atmosphere.set_elevation(new_altitude)

    def calculate_angles(self, collective_angle, lateral_angle, longitudinal_angle, velocity):
        # Compute rotor angles based on flight parameters
        

        # Inner functions for beta calculations
        def calculate_beta_1c(self, azimuth, **args):
            return (args['beta_0'] + args['beta_1c'] * np.cos(azimuth) + args['beta_1s'] * np.sin(azimuth)) * np.cos(azimuth)

        def calculate_beta_1s(self, azimuth, **args):
            return (args['beta_0'] + args['beta_1c'] * np.cos(azimuth) + args['beta_1s'] * np.sin(azimuth)) * np.sin(azimuth)

        # Function to solve for beta values
        self.main_rotor.beta_0 = 0
        def optimization_function(vars):
            beta_1c, beta_1s = vars
            self.main_rotor.beta_1c = beta_1c
            self.main_rotor.beta_1s = beta_1s
            beta_0 = self.main_rotor.estimate_total_beta_0(collective_angle, longitudinal_angle, lateral_angle, velocity, self.atmosphere.density,
                                                              self.main_rotor.beta_0 if hasattr(self.main_rotor, 'beta_0') else 0,
                                                              beta_1c, beta_1s)
            self.main_rotor.beta_0 = beta_0
            beta_1c_assumed = self.main_rotor.numerical_integrate(calculate_beta_1c, 0, 2 * np.pi, 100, beta_0=beta_0, beta_1c=beta_1c, beta_1s=beta_1s)
            beta_1s_assumed = self.main_rotor.numerical_integrate(calculate_beta_1s, 0, 2 * np.pi, 100, beta_0=beta_0, beta_1c=beta_1c, beta_1s=beta_1s)

            # Define equations to solve
            equation_1 = beta_1s - beta_1s_assumed
            equation_2 = beta_1c - beta_1c_assumed
            return [equation_1, equation_2]

        # Use fsolve to determine optimal beta values
        beta_1c, beta_1s = fsolve(optimization_function, [0.001, 0.001], xtol=0.0005)

        beta_0 = self.main_rotor.estimate_total_beta_0(collective_angle, longitudinal_angle, lateral_angle, velocity, self.atmosphere.density,
                                                              self.main_rotor.beta_0 if hasattr(self.main_rotor, 'beta_0') else 0,
                                                              beta_1c, beta_1s)

        # Assign computed beta values to the rotor
        self.main_rotor.beta_0 = beta_0
        self.main_rotor.beta_1s = beta_1s
        self.main_rotor.beta_1c = beta_1c 

        # Return the estimated beta values
        return beta_0, beta_1s, beta_1c 

    def calculate_total_power(self, theta_0, theta_1s=0, theta_1c=0, V=0, theta_tail=0):
        # Compute the total power required for the helicopter
        if self.flight_mode == 'hover':
            # In hover mode, calculate the main rotor power
            main_power = self.calculate_power(theta_0, theta_1s, theta_1c, V)
        elif self.flight_mode == 'forward':
            # In forward flight, calculate the power for each rotor blade and sum them
            main_power = np.sum([self.main_rotor.compute_power_in_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) 
                                for blade in self.main_rotor.blade_array])
        
        # Get the ideal tail rotor power and angle for the current flight condition
        tail_power = self.calculate_tail_power(theta_tail)

        # Calculate total power: main power + tail power + drag from the fuselage
        V = V if self.flight_mode == 'forward' else 0
        total_power = main_power + tail_power + self.get_fuselage_drag(V) * V
        
        # Return total power, main rotor power, and tail rotor power
        return total_power, main_power, tail_power

    def calculate_power(self, theta_0, theta_1s=0, theta_1c=0, V=0):
        # Calculate the power of the helicopter based on flight conditions and rotor angles
        if self.flight_mode == 'hover':
            # In hover mode, compute power for the main rotor in hover
            power = self.main_rotor.compute_hover_power(theta_0, self.atmosphere.density, V)
        elif self.flight_mode == 'forward':
            # In forward mode, calculate induced power from all rotor blade_array
            induced_power = np.sum([self.main_rotor.get_induced_power_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) 
                                    for blade in self.main_rotor.blade_array])
            
            # Calculate profile power for all rotor blade_array
            profile_power = np.sum([self.main_rotor.get_profile_power_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) 
                                    for blade in self.main_rotor.blade_array])
            
            # Total power is the sum of induced and profile power
            power = induced_power + profile_power
            
        # Calculate drag on the fuselage based on velocity and drag coefficient
        drag_fuselage = 1/2 * self.atmosphere.density * V**2 * np.pi * self.main_rotor_cg_z**2 * self.fuselage_drag_coefficient if self.flight_mode == 'forward' else 0
            
        # Return total power (including drag) and separate power components
        return power + drag_fuselage * V

    def calculate_thrust(self, theta_0, theta_1s=0, theta_1c=0, V=0):
        # Calculate the thrust produced by the helicopter's rotors
        if self.flight_mode == 'hover':
            # In hover mode, calculate thrust from the main rotor
            thrust = self.main_rotor.compute_hover_thrust(theta_0, self.atmosphere.density, V)
        elif self.flight_mode == 'forward':
            # In forward mode, calculate thrust from all rotor blade_array
            thrust = np.sum([self.main_rotor.compute_thrust_in_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) 
                            for blade in self.main_rotor.blade_array])
            drag = self.calculate_total_drag(theta_0, theta_1c, theta_1s, V)
            angle_f = np.arctan(drag/thrust)
            thrust += 0.5 * self.atmosphere.density * V**2 * self.horizontal_stabilizer.chord * self.horizontal_stabilizer.span * self.horizontal_stabilizer.calculate_lift_coefficient(-angle_f)
        
        # Update previous thrust coefficient (Ct) based on calculated thrust
        self.main_rotor.previous_ct = thrust / (self.atmosphere.density * np.pi * self.main_rotor.blade_array[0].radius**4 * (self.main_rotor.angular_velocity**2))
        
        return thrust

    def calculate_torque(self, theta_0, theta_1s=0, theta_1c=0, V=0):
        # Calculate the torque produced by the helicopter's rotors
        if self.flight_mode == 'hover':
            # In hover mode, calculate torque from the main rotor
            torque = self.main_rotor.calculate_torque_hover(theta_0, self.atmosphere.density, V)
        elif self.flight_mode == 'forward':
            # In forward mode, calculate torque from all rotor blade_array
            torque = np.sum([self.main_rotor.compute_torque_in_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) 
                            for blade in self.main_rotor.blade_array])
        
        return torque
    
    def calculate_total_drag(self, theta_0, theta_1c, theta_1s, V):
        if self.flight_mode == 'hover':
            return 0
        
        drag = np.sum([self.main_rotor.compute_drag_in_forward_flight(blade, theta_0, theta_1s, theta_1c, V, self.atmosphere.density) for blade in self.main_rotor.blade_array])
        drag_fuselage = self.get_fuselage_drag(V)
        
        angle_f = np.arctan((drag + drag_fuselage) / self.total_weight)
        hs_drag = 0.5 * self.atmosphere.density * V**2 * self.horizontal_stabilizer.chord * self.horizontal_stabilizer.span * self.horizontal_stabilizer.calculate_drag_coefficient(-angle_f)
        vs_drag = 0.5 * self.atmosphere.density * V**2 * self.vertical_stabilizer.chord * self.vertical_stabilizer.span * self.vertical_stabilizer.calculate_drag_coefficient(0)
   
        return drag + drag_fuselage + hs_drag + vs_drag
    
    def get_fuselage_drag(self, V):
        # Calculate the drag force on the fuselage based on velocity and drag coefficient
        return 1/2 * self.atmosphere.density * V**2 * np.pi * self.main_rotor_cg_z**2 * self.fuselage_drag_coefficient

    def calculate_tail_power(self, theta_0, theta_1s=0, theta_1c=0, V=0):
        # Calculate the power required by the tail rotor during hover
        return self.tail_rotor.compute_hover_power(theta_0, self.atmosphere.density, 0)

    def calculate_tail_thrust(self, theta_0, theta_1s=0, theta_1c=0, V=0):
        # Calculate the thrust produced by the tail rotor during hover
        return self.tail_rotor.compute_hover_thrust(theta_0, self.atmosphere.density, 0)

    def initialize_simulation(self, collective_pitch, lateral_pitch, longitudinal_pitch, V=None, timesteps=1000, checkpoints=[0, 50, 200]):
        # setup simulation for a given number of timesteps, estimating angles at specified checkpoints
        if V is None:
            V = self.velocity
        for i in range(timesteps):
            if i in checkpoints:
                start_time = time.time()  # Start timer to measure computation time
                self.calculate_angles(collective_pitch, longitudinal_pitch, lateral_pitch, V)

                # Calculate thrust at each timestep
                self.calculate_thrust(collective_pitch, longitudinal_pitch, lateral_pitch, V)


class MissionCoordinator:

    def __init__(self):
        self.fuel_energy_content = 43.124 * 1e6  # J/kg

        data_frame = pd.read_csv('engine_data.csv')

        data_frame = data_frame[data_frame['Delta T from ISA (K)'] == 0].reset_index(drop=True).drop(columns=['Delta T from ISA (K)'])
        altitudes = np.arange(0, 3001, 100)
        self.engine_performance = pd.DataFrame({
            "Altitude (m)": altitudes,
            "Prop Shaft Power Delivered (kW)": np.interp(altitudes, data_frame["Altitude (m)"], data_frame["Prop Shaft Power Delivered (kW)"]),
            "SFC (kg/kWh)": np.interp(altitudes, data_frame["Altitude (m)"], data_frame["SFC (kg/kWh)"]),
            "Fuel Flow (L/h)": np.interp(altitudes, data_frame["Altitude (m)"], data_frame["Fuel Flow (L/h)"])
        })

    def retrieve_engine_power(self, altitude):
        altitude = int(math.ceil(altitude / 100.0)) * 100
        return self.engine_performance.loc[self.engine_performance['Altitude (m)'] == altitude, 'Prop Shaft Power Delivered (kW)'].values[0]

    def retrieve_sfc(self, altitude):
        altitude = int(math.ceil(altitude / 100.0)) * 100
        return self.engine_performance.loc[self.engine_performance['Altitude (m)'] == altitude, 'SFC (kg/kWh)'].values[0]

    def is_fuel_exceeded(self):
        return self.fuel_weight < 0

    def is_power_exceeded(self, power, available_power):
        return power > available_power * 1000

    def generate_estimator_functions(self, helicopter: HelicopterModel, climb_velocity, flight_conditions):
        helicopter.flight_mode = 'hover'
        angles = np.radians(np.arange(15))

        # Calculate thrust and power for hover
        thrust_values = helicopter.calculate_thrust(angles)
        power_values = helicopter.calculate_total_power(angles)[0]
        coefficients = np.polyfit(thrust_values, power_values, 2)
        hover_function = np.poly1d(coefficients)

        rotor_radius = helicopter.main_rotor.blade_array[0].radius
        rotor_speed = helicopter.main_rotor.angular_velocity
        solidity = helicopter.main_rotor.blade_count * helicopter.main_rotor.get_average_chord() / (np.pi * rotor_radius)

        thrust_data = [(helicopter.calculate_thrust(t, V=climb_velocity), climb_velocity) for t in angles]
        power_data = [helicopter.calculate_total_power(t, V=climb_velocity)[0] for t in angles]

        x1 = fsolve(lambda vars: sum(abs(p - t[0] * (t[1] / 2 + np.sqrt((t[1] / 2) ** 2 + t[0] / (2 * helicopter.atmosphere.density * np.pi * rotor_radius ** 2))) - vars[0] * t[0])
                                        for t, p in zip(thrust_data, power_data)), [0])[0]

        vertical_ascent_function = lambda t, v: t * (v / 2 + np.sqrt((v / 2) ** 2 + t / (2 * helicopter.atmosphere.density * np.pi * rotor_radius ** 2))) + x1 * t

        helicopter.flight_mode = 'forward'

        power_list, thrust_list, drag_list, lambda_list = [], [], [], []

        for condition in flight_conditions:
            helicopter.update_altitude(condition['alt'])
            helicopter.V = condition['V']
            helicopter.gross_weight = condition['weight']
            helicopter.initialize_simulation(condition['theta_0'], condition['theta_1c'], condition['theta_1s'], condition['V'], timesteps=200, checkpoints=[0, 50])

            thrust = helicopter.calculate_thrust(condition['theta_0'], condition['theta_1s'], condition['theta_1c'], condition['V'])
            power = helicopter.calculate_total_power(condition['theta_0'], condition['theta_1s'], condition['theta_1c'], condition['V'], theta_tail=condition['theta_tail'])[0]

            power_list.append(power)
            thrust_list.append(thrust)
            drag_list.append(helicopter.get_fuselage_drag(condition['V']) * condition['V'])

            thrust_coefficient = thrust / (helicopter.atmosphere.density * np.pi * rotor_radius ** 4 * (helicopter.main_rotor.angular_velocity ** 2))
            mu = condition['V'] / (helicopter.main_rotor.angular_velocity * rotor_radius)

            lambda_gi = fsolve(lambda vars: vars - thrust_coefficient / (2 * np.sqrt(mu ** 2 + (vars[0]) ** 2)), [0.001])[0]
            lambda_list.append(lambda_gi)

        x2 = fsolve(lambda vars: sum(p - t * lambd * helicopter.main_rotor.angular_velocity * rotor_radius - fd - vars[0] * (helicopter.main_rotor.angular_velocity ** 2 * rotor_radius ** 2 + 4.6 * condition['V'] ** 2)
                                    for t, lambd, fd, condition, p in zip(thrust_list, lambda_list, drag_list, flight_conditions, power_list)), [0])[0]

        def level_flight(thrust_required, V):
            thrust_coefficient = thrust_required / (helicopter.atmosphere.density * np.pi * rotor_radius ** 4 * (helicopter.main_rotor.angular_velocity ** 2))
            mu = V / (helicopter.main_rotor.angular_velocity * rotor_radius)

            lambda_gi = fsolve(lambda vars: vars - thrust_coefficient / (2 * np.sqrt(mu ** 2 + (vars[0]) ** 2)), [0.001])[0]
            drag_force = helicopter.get_fuselage_drag(V) * V

            return thrust_required * lambda_gi * rotor_speed * rotor_radius + drag_force + x2 * (rotor_speed ** 2 * rotor_radius ** 2 + 4.6 * V ** 2)

        return hover_function, vertical_ascent_function, hover_function, level_flight

    def evaluate_mission_parameters(self, mission, inputs, forward_flight_conditions):
        def update_mission_status(helicopter, power, speed_val, climb_rate_val, distance, time_interval=5):
            power_list.append(power)
            fuel_used = self.retrieve_sfc(altitudes[-1]) * power / 1000 * time_interval / 3600
            self.fuel_weight -= fuel_used
            self.gross_weight -= fuel_used
            fuel_weight_list.append(self.fuel_weight)
            gross_weight_list.append(self.gross_weight)
            fuel_burn_rate_list.append(fuel_used / time_interval)

            altitudes.append(helicopter.atmosphere.elevation)
            available_power_list.append(self.retrieve_engine_power(altitudes[-1]))
            speed_list.append(speed_val)
            climb_rate_list.append(climb_rate_val)
            distance_covered_list.append(distance_covered_list[-1] + distance)
            time_list.append(time_list[-1] + time_interval)

            if self.is_fuel_exceeded() or self.is_power_exceeded(power_list[-1], available_power_list[-1]):
                print("\033[91m Limit Exceeded! Mission can't be completed \033[0m")
                return True
            return False
        parameters = mission['parameters']
        current_altitude, power_list, available_power_list = 0, [], []
        gross_weight_list, fuel_weight_list, fuel_burn_rate_list = [], [], []
        altitudes, speed_list, climb_rate_list, distance_covered_list, time_list = [2000], [], [], [0], [0]

        # Define each mission type
        for mission_type, values in mission['mission'].items():
            if 'takeoff' in mission_type:
                helicopter = self.handle_takeoff(values, inputs, parameters, forward_flight_conditions)
            elif 'vertical_climb' in mission_type:
                self.handle_vertical_climb(values, helicopter, parameters, current_altitude, update_mission_status)
                current_altitude = values['alt']  # Update current altitude after climbing
            elif 'steady_climb' in mission_type:
                current_altitude = self.handle_steady_climb(values, helicopter, parameters, current_altitude, update_mission_status)
            elif 'level_flight' in mission_type:
                self.handle_level_flight(values, helicopter, parameters, update_mission_status)
            elif 'steady_descent' in mission_type:
                current_altitude = self.handle_steady_descent(values, helicopter, parameters, current_altitude, update_mission_status)
            elif 'vertical_descent' in mission_type:
                self.handle_vertical_descent(values, helicopter, parameters, current_altitude, update_mission_status)
            elif 'hover' in mission_type:
                self.handle_hover(values, helicopter, update_mission_status)
            elif 'payload_drop' in mission_type:
                self.handle_payload_drop(values, helicopter)
            elif 'payload_pickup' in mission_type:
                self.handle_payload_pickup(values, helicopter)

        # Return final results
        return {'gross_weight': gross_weight_list, 'fuel_weight': fuel_weight_list, 'fuel_burn_rate': fuel_burn_rate_list, 'altitudes': altitudes, 'powers': power_list, 'available_engine_power': available_power_list, 'speed': speed_list, 'climb_rate': climb_rate_list, 'distance_covered': distance_covered_list, 'time': time_list}

    def handle_takeoff(self, values, inputs, parameters, forward_flight_conditions):
        inputs['helicopter']['payload'], inputs['helicopter']['fuel'] = values['payload'], values['fuel']
        inputs['mission_planner'].update({'payload': values['payload'], 'fuel': values['fuel'], 'takeoff_alt': values['alt']})
        payload, fuel, current_altitude = values['payload'], values['fuel'], values['alt']
        helicopter, self.payload, self.fuel_weight = HelicopterModel(inputs), payload, fuel
        self.body_weight = inputs['mission_planner']['body_weight']
        self.gross_weight = self.payload + self.fuel_weight + self.body_weight
        self.hover_function, self.vertical_ascent_function, self.vertical_descent_function, self.level_flight_function = self.generate_estimator_functions(
            helicopter, parameters['vertical_climb_speed'], forward_flight_conditions
        )
        return helicopter

    def handle_vertical_climb(self, values, helicopter, parameters, current_altitude, update_mission_status):
        target_altitude = values['alt']
        time_needed = (target_altitude - current_altitude) / parameters['vertical_climb_speed']
        thrust_needed = self.gross_weight * 10
        for t in np.arange(0, time_needed, 5 if time_needed > 5 else 1):
            power = self.vertical_ascent_function(thrust_needed, parameters['vertical_climb_speed'])
            if update_mission_status(helicopter, power, 0, parameters['vertical_climb_speed'], 0):
                return current_altitude  # Return current altitude if mission is aborted

    def handle_steady_climb(self, values, helicopter, parameters, current_altitude, update_mission_status):
        target_altitude, tailwind = values['alt'], values['tailwind'] * 5 / 18
        time_needed = (target_altitude - current_altitude) / parameters['steady_climb_upward_speed']
        overall_speed = parameters['steady_climb_forward_speed'] + tailwind
        thrust_needed = self.gross_weight * 10

        for t in np.arange(0, time_needed, 5 if time_needed > 5 else 1):
            power = self.level_flight_function(thrust_needed, parameters['steady_climb_forward_speed']) + \
                    self.gross_weight * 10 * parameters['steady_climb_upward_speed'] * 5
            if update_mission_status(helicopter, power, overall_speed, parameters['steady_climb_upward_speed'], overall_speed * (5 if time_needed > 5 else 1)):
                return current_altitude  # Return current altitude if mission is aborted
        return target_altitude  # Return the new altitude after climbing

    def handle_level_flight(self, values, helicopter, parameters, update_mission_status):
        distance, tailwind = values['distance'], values['tailwind'] * 5 / 18
        overall_speed = parameters['level_flight_speed'] + tailwind
        time_needed = distance / overall_speed
        thrust_needed = self.gross_weight * 10

        for t in np.arange(0, time_needed, 5):
            power = self.level_flight_function(thrust_needed, parameters['level_flight_speed'])
            if update_mission_status(helicopter, power, overall_speed, 0, overall_speed * 5):
                return  # Abort mission if limits are exceeded

    def handle_steady_descent(self, values, helicopter, parameters, current_altitude, update_mission_status):
        target_altitude, tailwind = values['alt'], values['tailwind'] * 5 / 18
        time_needed = abs((current_altitude - target_altitude) / parameters['steady_descent_downward_speed'])
        overall_speed = parameters['steady_descent_forward_speed'] + tailwind
        thrust_needed = self.gross_weight * 10

        for t in np.arange(0, time_needed, 5 if time_needed > 5 else 1):
            power = self.level_flight_function(thrust_needed, abs(parameters['steady_descent_forward_speed'])) + \
                    self.gross_weight * 10 * parameters['steady_descent_downward_speed'] * 5
            if update_mission_status(helicopter, power, overall_speed, parameters['steady_descent_downward_speed'], overall_speed * (5 if time_needed > 5 else 1)):
                return current_altitude  # Return current altitude if mission is aborted
        return target_altitude  # Return the new altitude after descent

    def handle_vertical_descent(self, values, helicopter, parameters, current_altitude, update_mission_status):
        target_altitude = values['alt']
        time_needed = abs((current_altitude - target_altitude) / parameters['vertical_descent_speed'])
        thrust_needed = self.gross_weight * 10
        for t in np.arange(0, time_needed, 5 if time_needed > 5 else 1):
            power = self.vertical_descent_function(thrust_needed)
            if update_mission_status(helicopter, power, 0, parameters['vertical_descent_speed'], 0):
                return  # Abort mission if limits are exceeded

    def handle_hover(self, values, helicopter, update_mission_status):
        for t in np.arange(0, values['time'], 5):
            power = self.hover_function(self.gross_weight * 10)
            if update_mission_status(helicopter, power, 0, 0, 0):
                return  # Abort mission if limits are exceeded

    def handle_payload_drop(self, values, helicopter):
        drop_weight = values['weight_to_drop']
        if self.payload < drop_weight:
            raise Exception('Payload to drop exceeds available payload')
        else:
            self.payload -= drop_weight
            self.gross_weight -= drop_weight

    def handle_payload_pickup(self, values, helicopter):
        self.payload += values['weight_to_pickup']
        self.gross_weight += values['weight_to_pickup']

  

class RotorSystem:  # Renamed from Rotor
    
    def __init__(self, config):
        # Create blade array with evenly distributed azimuth angles
        angle_step = 360 / config['num_blades']
        self.blade_array = [Blade(config, azimuth=i * np.pi / 180) 
                           for i in np.arange(0, 360, angle_step)]
        
        # System properties
        self.blade_count = config['num_blades']
        self.tip_path_plane_angle = 0
        self.operating_state = 'stationary'  # Changed from 'mode'
        
        # Physical properties
        self.moment_of_inertia = 1/3 * 0.4 * self.blade_array[0].radius ** 2
        self.attack_angle = 0
        self.thrust_coefficient_previous = 0.0017

    def numerical_integrate(self, function, lower_bound, upper_bound, segments=100, **kwargs):
        # Enhanced numerical integration using midpoint method
        segment_width = (upper_bound - lower_bound) / segments
        points = np.arange(lower_bound, upper_bound, segment_width)
        result = 0
        
        for x in points:
            midpoint = x + segment_width / 2
            result += function(self, midpoint, **kwargs) * segment_width
            
        return result

    def check_reverse_flow(self, tangential_velocity):
        return tangential_velocity < 0

    def compute_inflow_distribution(self, radial_pos, blade_radius, forward_speed, plane_angle, azimuth):
        # Calculate non-uniform inflow distribution with enhanced methodology
        advance_ratio_axial = forward_speed * np.cos(plane_angle) / (self.angular_velocity * self.blade_array[0].radius)
        advance_ratio_vertical = forward_speed * np.sin(plane_angle) / (self.angular_velocity * self.blade_array[0].radius)
        
        def inflow_equation(vars):
            induced_velocity = vars
            return (induced_velocity - self.thrust_coefficient_previous / 
                   (2 * np.sqrt(advance_ratio_axial ** 2 + 
                               (advance_ratio_vertical + induced_velocity) ** 2)))
        
        induced_velocity_initial = fsolve(inflow_equation, [0.001])[0]
        total_inflow = induced_velocity_initial + advance_ratio_vertical
        
        # Enhanced inflow model with improved non-uniformity factor
        correction_factor = ((4/3 * advance_ratio_axial / total_inflow) / 
                           (1.2 + advance_ratio_axial / total_inflow))
        
        local_induced_velocity = induced_velocity_initial * (1 + correction_factor * 
                                                           radial_pos / blade_radius * 
                                                           np.cos(azimuth))
        
        return local_induced_velocity
    
    def compute_thrust_in_forward_flight(self, blade, pitch_collective, pitch_lateral, pitch_longitudinal, 
                                       airspeed, air_density, **kwargs):
        """Calculate thrust forces during forward flight conditions."""
        beta_0, beta_1s, beta_1c = (self.beta_0, self.beta_1s, 
                                                     self.beta_1c)

        def thrust_distribution(self, radius, **params):
            """Compute local thrust contribution at given radius."""
            chord = params['blade'].calculate_chord(radius)
            lift_slope = params['blade'].lift_coefficient_slope
            azimuth = params['azimuth']
            
            # Total pitch angle calculation
            total_pitch = (params['pitch_collective'] + 
                         blade.calculate_twist(radius) + 
                         params['pitch_longitudinal'] * np.cos(azimuth) + 
                         params['pitch_lateral'] * np.sin(azimuth))

            # Flow field calculations
            inflow = self.compute_inflow_distribution(radius, params['blade'].radius, 
                                                    airspeed, params['tip_plane_angle'], 
                                                    params['azimuth'])
            
            # Velocity components
            vel_tangential = (self.angular_velocity * radius + 
                            airspeed * np.cos(params['tip_plane_angle']) * np.sin(azimuth))
            vel_perpendicular = (self.angular_velocity * params['blade'].radius * inflow + 
                               airspeed * np.sin(params['tip_plane_angle']) + 
                               airspeed * np.sin(params['flap']) * np.cos(azimuth) + 
                               radius * params['flap_rate'])
            
            if self.check_reverse_flow(vel_tangential):
                return 0
            
            # Local thrust calculation with improved aerodynamic model
            return ((0.5 * params['air_density'] * chord * lift_slope * 
                    (vel_tangential ** 2 + vel_perpendicular ** 2) * 
                    (total_pitch - np.arctan(vel_perpendicular / vel_tangential))) * 
                    np.cos(params['flap']))

        # Compute flapping motion parameters
        azimuth = blade.azimuth
        flap = (beta_0 + 
                beta_1c * np.cos(azimuth) + 
                beta_1s * np.sin(azimuth))
        flap_rate = self.angular_velocity * (-beta_1c * np.sin(blade.azimuth) + 
                                           beta_1s * np.cos(blade.azimuth))
        tip_plane_angle = (self.attack_angle + 
                          beta_1c * np.cos(blade.azimuth) + 
                          beta_1s * np.sin(blade.azimuth))

        # Integrate thrust over blade span
        return self.numerical_integrate(
            thrust_distribution, 
            blade.root_cutout, 
            blade.radius,
            pitch_collective=pitch_collective,
            pitch_lateral=pitch_lateral,
            pitch_longitudinal=pitch_longitudinal,
            blade=blade,
            air_density=air_density,
            airspeed=airspeed,
            flap=flap,
            azimuth=azimuth,
            tip_plane_angle=tip_plane_angle,
            flap_rate=flap_rate
        )

    def compute_drag_in_forward_flight(self, blade, pitch_collective, pitch_lateral, 
                                     pitch_longitudinal, airspeed, air_density, **kwargs):
        """Calculate aerodynamic drag during forward flight."""
        beta_0, beta_1s, beta_1c = (self.beta_0, self.beta_1s, 
                                                     self.beta_1c)

        def drag_distribution(self, radius, **params):
            """Compute local drag contribution at given radius."""
            chord = params['blade'].calculate_chord(radius)
            lift_slope = params['blade'].lift_coefficient_slope
            
            # Total pitch calculation with all components
            total_pitch = (params['pitch_collective'] + 
                         blade.calculate_twist(radius) + 
                         params['pitch_longitudinal'] * np.cos(blade.azimuth) + 
                         params['pitch_lateral'] * np.sin(blade.azimuth))

            # Enhanced flow field calculations
            inflow = self.compute_inflow_distribution(radius, params['blade'].radius, 
                                                    airspeed, params['tip_plane_angle'], 
                                                    params['azimuth'])
            
            # Velocity components with improved modeling
            vel_tangential = (self.angular_velocity * radius + 
                            airspeed * np.cos(params['tip_plane_angle']) * 
                            np.sin(params['azimuth']))
            vel_perpendicular = (self.angular_velocity * params['blade'].radius * inflow + 
                               airspeed * np.sin(params['tip_plane_angle']) + 
                               airspeed * np.sin(params['flap']) * np.cos(params['azimuth']) + 
                               radius * params['flap_rate'])
            
            if self.check_reverse_flow(vel_tangential):
                return (0.5 * params['air_density'] * (vel_tangential ** 2 + vel_perpendicular ** 2) * 
                       1.28 * (chord * np.sin(total_pitch - np.arctan(vel_perpendicular / vel_tangential))))
            
            # Advanced aerodynamic calculations
            effective_angle = total_pitch - np.arctan(vel_perpendicular / vel_tangential)
            profile_drag = (0.5 * params['air_density'] * (vel_tangential ** 2 + vel_perpendicular ** 2) * 
                          chord * params['blade'].calculate_drag_coefficient(effective_angle))
            induced_drag = ((0.5 * params['air_density'] * chord * lift_slope * 
                           (vel_tangential ** 2 + vel_perpendicular ** 2) * 
                           effective_angle) * (vel_perpendicular / vel_tangential))
            
            return profile_drag + induced_drag

        # Compute flapping parameters
        azimuth = blade.azimuth
        flap = (beta_0 + 
                beta_1c * np.cos(azimuth) + 
                beta_1s * np.sin(azimuth))
        flap_rate = self.angular_velocity * (-beta_1c * np.sin(blade.azimuth) + 
                                           beta_1s * np.cos(blade.azimuth))
        tip_plane_angle = (self.attack_angle + 
                          beta_1c * np.cos(blade.azimuth) + 
                          beta_1s * np.sin(blade.azimuth))

        # Integrate drag over blade span
        return self.numerical_integrate(
            drag_distribution,
            blade.root_cutout,
            blade.radius,
            pitch_collective=pitch_collective,
            pitch_lateral=pitch_lateral,
            pitch_longitudinal=pitch_longitudinal,
            blade=blade,
            air_density=air_density,
            airspeed=airspeed,
            flap=flap,
            azimuth=azimuth,
            tip_plane_angle=tip_plane_angle,
            flap_rate=flap_rate
        )
    
    def compute_torque_in_forward_flight(self, blade, pitch_collective, pitch_lateral, 
                                       pitch_longitudinal, airspeed, air_density, **kwargs):
        """Calculate rotor torque during forward flight conditions."""
        beta_0, beta_1s, beta_1c = (self.beta_0, self.beta_1s, 
                                                     self.beta_1c)

        def torque_distribution(self, radius, **params):
            """Calculate local torque contribution at specific radius."""
            chord = params['blade'].calculate_chord(radius)
            lift_slope = params['blade'].lift_coefficient_slope
            
            # Calculate total blade pitch angle
            total_pitch = (params['pitch_collective'] + 
                         blade.calculate_twist(radius) + 
                         params['pitch_longitudinal'] * np.cos(blade.azimuth) + 
                         params['pitch_lateral'] * np.sin(blade.azimuth))

            # Advanced inflow modeling
            inflow = self.compute_inflow_distribution(radius, params['blade'].radius, 
                                                    airspeed, params['tip_plane_angle'], 
                                                    params['azimuth'])
            
            # Enhanced velocity calculations
            vel_tangential = (self.angular_velocity * radius + 
                            airspeed * np.cos(params['tip_plane_angle']) * 
                            np.sin(params['azimuth']))
            vel_perpendicular = (self.angular_velocity * params['blade'].radius * inflow + 
                               airspeed * np.sin(params['tip_plane_angle']) + 
                               airspeed * np.sin(params['flap']) * np.cos(params['azimuth']) + 
                               radius * params['flap_rate'])

            # Handle reverse flow conditions
            if self.check_reverse_flow(vel_tangential):
                return ((0.5 * params['air_density'] * (vel_tangential**2 + vel_perpendicular**2) * 
                        1.28 * chord * np.sin(total_pitch - np.arctan(vel_perpendicular / vel_tangential))) * radius)

            # Advanced torque calculations
            return ((0.5 * params['air_density'] * (vel_tangential**2 + vel_perpendicular**2) * chord * 
                    params['blade'].calculate_drag_coefficient(total_pitch - np.arctan(vel_perpendicular / vel_tangential))) * 
                    radius)

        # Calculate flapping parameters
        azimuth = blade.azimuth
        flap = (beta_0 + 
                beta_1c * np.cos(azimuth) + 
                beta_1s * np.sin(azimuth))
        flap_rate = self.angular_velocity * (-beta_1c * np.sin(azimuth) + 
                                           beta_1s * np.cos(azimuth))
        tip_plane_angle = (self.attack_angle + 
                          beta_1c * np.cos(azimuth) + 
                          beta_1s * np.sin(azimuth))

        # Integrate torque over blade span
        return self.numerical_integrate(
            torque_distribution,
            blade.root_cutout,
            blade.radius,
            pitch_collective=pitch_collective,
            pitch_lateral=pitch_lateral,
            pitch_longitudinal=pitch_longitudinal,
            blade=blade,
            air_density=air_density,
            airspeed=airspeed,
            flap=flap,
            azimuth=azimuth,
            tip_plane_angle=tip_plane_angle,
            flap_rate=flap_rate
        )
    
    def compute_induced_power_forward(self, blade, collective_pitch, cyclic_lateral, 
                                    cyclic_longitudinal, velocity, air_density, **kwargs):
        """Calculate induced power during forward flight using enhanced blade element theory."""
        
        # Get flapping coefficients
        beta_0, flap_lat, flap_long = self.beta_0, self.beta_1s, self.beta_1c

        def induced_power_element(self, radius, **params):
            """Compute local induced power contribution at blade element."""
            # Get blade geometry parameters
            chord_length = params['blade'].calculate_chord(radius)
            lift_slope = params['blade'].lift_coefficient_slope
            
            # Calculate total pitch including all components
            total_pitch = (params['collective'] + 
                         blade.calculate_twist(radius) + 
                         params['cyclic_long'] * np.cos(blade.azimuth) + 
                         params['cyclic_lat'] * np.sin(blade.azimuth))

            # Enhanced aerodynamic calculations
            local_inflow = self.compute_inflow_distribution(
                radius, params['blade'].radius, 
                params['velocity'], 
                params['tip_plane_angle'], 
                params['azimuth']
            )
            
            # Advanced velocity calculations
            tangential_vel = (self.angular_velocity * radius + 
                            params['velocity'] * np.cos(params['tip_plane_angle']) * 
                            np.sin(params['azimuth']))
            
            perpendicular_vel = (self.angular_velocity * params['blade'].radius * local_inflow + 
                               params['velocity'] * np.sin(params['tip_plane_angle']) + 
                               params['velocity'] * np.sin(params['flap']) * 
                               np.cos(params['azimuth']) + 
                               radius * params['flap_rate'])

            # Handle reverse flow regions
            if self.check_reverse_flow(tangential_vel):
                return 0
            
            # Calculate effective angle and induced power
            effective_angle = total_pitch - np.arctan(perpendicular_vel / tangential_vel)
            return ((0.5 * params['air_density'] * chord_length * lift_slope * 
                    (tangential_vel**2 + perpendicular_vel**2) * 
                    effective_angle) * perpendicular_vel)
            
        # Calculate flapping motion
        azimuth = blade.azimuth
        flap = (beta_0 + 
                flap_long * np.cos(azimuth) + 
                flap_lat * np.sin(azimuth))
        flap_rate = self.angular_velocity * (-flap_long * np.sin(azimuth) + 
                                           flap_lat * np.cos(azimuth))
        tip_plane_angle = (self.attack_angle + 
                          flap_long * np.cos(azimuth) + 
                          flap_lat * np.sin(azimuth))

        # Compute total induced power through integration
        return self.numerical_integrate(
            induced_power_element,
            blade.root_cutout,
            blade.radius,
            collective=collective_pitch,
            cyclic_lat=cyclic_lateral,
            cyclic_long=cyclic_longitudinal,
            blade=blade,
            air_density=air_density,
            velocity=velocity,
            flap=flap,
            azimuth=azimuth,
            tip_plane_angle=tip_plane_angle,
            flap_rate=flap_rate
        )

    def compute_profile_power_forward(self, blade, collective_pitch, cyclic_lateral, 
                                    cyclic_longitudinal, velocity, air_density, **kwargs):
        """Calculate profile power during forward flight using advanced aerodynamic modeling."""
        
        # Get flapping coefficients
        beta_0, flap_lat, flap_long = self.beta_0, self.beta_1s, self.beta_1c

        def profile_power_element(self, radius, **params):
            """Compute local profile power contribution at blade element."""
            # Get blade geometry
            chord_length = params['blade'].calculate_chord(radius)
            lift_slope = params['blade'].lift_coefficient_slope
            
            # Total pitch calculation
            total_pitch = (params['collective'] + 
                         blade.calculate_twist(radius) + 
                         params['cyclic_long'] * np.cos(blade.azimuth) + 
                         params['cyclic_lat'] * np.sin(blade.azimuth))

            # Advanced flow calculations
            local_inflow = self.compute_inflow_distribution(
                radius, params['blade'].radius,
                params['velocity'], 
                params['tip_plane_angle'],
                params['azimuth']
            )
            
            # Enhanced velocity components
            tangential_vel = (self.angular_velocity * radius + 
                            params['velocity'] * np.cos(params['tip_plane_angle']) * 
                            np.sin(params['azimuth']))
            
            perpendicular_vel = (self.angular_velocity * params['blade'].radius * local_inflow + 
                               params['velocity'] * np.sin(params['tip_plane_angle']) + 
                               params['velocity'] * np.sin(params['flap']) * 
                               np.cos(params['azimuth']) + 
                               radius * params['flap_rate'])

            # Reverse flow handling with enhanced model
            if self.check_reverse_flow(tangential_vel):
                effective_angle = total_pitch - np.arctan(perpendicular_vel / tangential_vel)
                return ((0.5 * params['air_density'] * 
                        (tangential_vel**2 + perpendicular_vel**2) * 
                        1.28 * chord_length * np.sin(effective_angle)) * 
                        tangential_vel)

            # Advanced profile power calculation
            effective_angle = total_pitch - np.arctan(perpendicular_vel / tangential_vel)
            return (0.5 * params['air_density'] * 
                   (tangential_vel**2 + perpendicular_vel**2) * 
                   chord_length * 
                   params['blade'].calculate_drag_coefficient(effective_angle) * 
                   tangential_vel)
            
        # Calculate flapping parameters
        azimuth = blade.azimuth
        flap = (beta_0 + 
                flap_long * np.cos(azimuth) + 
                flap_lat * np.sin(azimuth))
        flap_rate = self.angular_velocity * (-flap_long * np.sin(azimuth) + 
                                           flap_lat * np.cos(azimuth))
        tip_plane_angle = (self.attack_angle + 
                          flap_long * np.cos(azimuth) + 
                          flap_lat * np.sin(azimuth))

        # Compute total profile power through integration
        return self.numerical_integrate(
            profile_power_element,
            blade.root_cutout,
            blade.radius,
            collective=collective_pitch,
            cyclic_lat=cyclic_lateral,
            cyclic_long=cyclic_longitudinal,
            blade=blade,
            air_density=air_density,
            velocity=velocity,
            flap=flap,
            azimuth=azimuth,
            tip_plane_angle=tip_plane_angle,
            flap_rate=flap_rate
        )

    def compute_power_in_forward_flight(self, blade, collective_pitch, cyclic_lateral, 
                                  cyclic_longitudinal, velocity, air_density, **kwargs):
        """Calculate total power in forward flight by combining induced and profile power."""
        
        # Calculate induced power component
        induced_power = self.compute_induced_power_forward(
            blade, collective_pitch, cyclic_lateral, cyclic_longitudinal,
            velocity, air_density, **kwargs
        )
        
        # Calculate profile power component
        profile_power = self.compute_profile_power_forward(
            blade, collective_pitch, cyclic_lateral, cyclic_longitudinal,
            velocity, air_density, **kwargs
        )

        # Return total power with additional parasitic effects consideration
        return induced_power + profile_power
    
    def estimate_total_beta_0(self, theta_0, theta_1c, theta_1s, V, density, beta_0=None, beta_1c=None, beta_1s=None, **args):
        # Get the first blade object from the blade_array list
        blade = self.blade_array[0]
        
        # Initialize beta_0, beta_1c, and beta_1s to default values if not provided
        if beta_0 is None:
            beta_0 = self.beta_0
            
        if beta_1c is None:
            beta_1c = self.beta_1c
            
        if beta_1s is None:
            beta_1s = self.beta_1s

        # Define a function to calculate beta_0 for a given azimuthal angle azimuth
        def get_beta_0(self, azimuth, **args):
            # Calculate the total angle of attack (a_tpp) based on beta_1c and beta_1s
            a_tpp = args['beta_1c'] * np.cos(azimuth) + args['beta_1s'] * np.sin(azimuth)

            # Define an inner function to calculate the integral of beta_0 over the radius r
            def int_beta_0(self, r, **args):
                # Get the chord length of the blade at radius r
                c = args['blade'].calculate_chord(r)
                # Get the lift curve slope of the blade
                a = args['blade'].lift_coefficient_slope
                
                # Calculate the effective angle of attack (theta) at radius r
                theta = args['theta_0'] + blade.calculate_twist(r) + args['theta_1c'] * np.cos(args['azimuth']) + args['theta_1s'] * np.sin(args['azimuth'])

                # Calculate the non-uniform inflow (lambda) at radius r
                lambd = self.compute_inflow_distribution(r, args['blade'].radius, V, args['a_tpp'], args['azimuth'])

                # Calculate the tangential velocity (U_T) at radius r
                U_T = self.angular_velocity * r + V * np.cos(args['a_tpp']) * np.sin(args['azimuth'])
                # Calculate the normal velocity (U_P) at radius r
                U_P = (self.angular_velocity * args['blade'].radius * lambd + 
                        V * np.sin(args['a_tpp']) + 
                        V * np.sin(args['beta']) * np.cos(args['azimuth']) + 
                        r * args['beta_dot'])
                
                # Check for reverse flow condition
                if self.check_reverse_flow(U_T):
                    return 0
                
                # Return the contribution to beta_0 from the current radius r
                return (1/2 * args['density'] * c * a * (U_T**2 + U_P**2) * 
                        (theta - np.arctan(U_P / U_T))) * r
                
            # Calculate the effective beta at the current azimuthal angle azimuth
            beta = args['beta_0'] + args['beta_1c'] * np.cos(azimuth) + args['beta_1s'] * np.sin(azimuth)
            # Calculate the rate of change of beta with respect to azimuth (beta_dot)
            beta_dot = self.angular_velocity * (-args['beta_1c'] * np.sin(azimuth) + args['beta_1s'] * np.cos(azimuth))

            # Integrate to calculate beta_0 over the blade's radius
            beta_0 = self.numerical_integrate( 
                int_beta_0, blade.root_cutout, blade.radius, 
                theta_0=theta_0, theta_1s=theta_1s, theta_1c=theta_1c, 
                blade=blade, density=density, V=V, beta=beta, azimuth=azimuth, beta_dot=beta_dot, a_tpp=a_tpp
            )

            return beta_0
        
        # Integrate get_beta_0 over the full 0 to 2*pi range of azimuth
        return self.numerical_integrate(
            get_beta_0, 0, 2 * np.pi, 
            theta_0=theta_0, theta_1s=theta_1s, theta_1c=theta_1c, 
            blade=blade, density=density, V=V, beta_0=beta_0, beta_1c=beta_1c, beta_1s=beta_1s
        ) / (2 * np.pi * self.angular_velocity**2 * self.moment_of_inertia)

    def compute_hover_power(self, collective_pitch, air_density, induced_velocity, **kwargs):
        """Calculate rotor power in hover conditions using blade element theory."""
        total_power = 0
        
        def power_distribution(self, radius, **params):
            """Calculate power distribution along blade radius in hover."""
            blade = params['blade']
            collective = params['collective']
            
            # Calculate local solidity
            local_solidity = (self.blade_count * blade.calculate_chord(radius)) / (np.pi * blade.radius)
            
            # Calculate total pitch including twist
            total_pitch = collective + blade.calculate_twist(radius)
            
            # Calculate inflow characteristics
            inflow = self.compute_hover_inflow(radius, total_pitch, local_solidity, 
                                             blade, induced_velocity)
            
            # Calculate flow angles
            flow_angle = np.arctan(inflow * blade.radius / radius)
            
            # Calculate aerodynamic coefficients
            lift_coeff = blade.calculate_lift_coefficient(total_pitch - flow_angle)
            drag_coeff = blade.calculate_drag_coefficient(total_pitch - flow_angle)
            
            # Calculate local chord
            chord = blade.calculate_chord(radius)
            
            # Calculate power contribution
            return (self.angular_velocity * radius * 0.5 * air_density * 
                   ((self.angular_velocity * radius)**2 + 
                    (inflow * self.angular_velocity * blade.radius)**2) * chord * 
                   (drag_coeff * np.cos(flow_angle) + lift_coeff * np.sin(flow_angle)))

        # Sum power contributions from all blades
        for blade in self.blade_array:
            total_power += self.numerical_integrate(
                power_distribution,
                blade.root_cutout,
                blade.radius,
                collective=collective_pitch,
                blade=blade
            )

        return total_power

    def compute_hover_thrust(self, collective_pitch, air_density, induced_velocity, **kwargs):
        """Calculate rotor thrust in hover conditions."""
        total_thrust = 0
        
        def thrust_distribution(self, radius, **params):
            """Calculate thrust distribution along blade radius in hover."""
            blade = params['blade']
            collective = params['collective']
            
            # Calculate local solidity
            local_solidity = (self.blade_count * blade.calculate_chord(radius)) / (np.pi * blade.radius)
            
            # Calculate total pitch including twist
            total_pitch = collective + blade.calculate_twist(radius)
            
            # Calculate inflow characteristics
            inflow = self.compute_hover_inflow(radius, total_pitch, local_solidity, 
                                             blade, induced_velocity)
            
            # Calculate flow angles
            flow_angle = np.arctan(inflow * blade.radius / radius)
            
            # Calculate aerodynamic coefficients
            lift_coeff = blade.calculate_lift_coefficient(total_pitch - flow_angle)
            drag_coeff = blade.calculate_drag_coefficient(total_pitch - flow_angle)
            
            # Calculate local chord
            chord = blade.calculate_chord(radius)
            
            # Calculate thrust contribution
            return (0.5 * air_density * 
                   ((self.angular_velocity * radius)**2 + 
                    (inflow * self.angular_velocity * blade.radius)**2) * chord * 
                   (lift_coeff * np.cos(flow_angle) - drag_coeff * np.sin(flow_angle)))

        # Sum thrust contributions from all blades
        for blade in self.blade_array:
            total_thrust += self.numerical_integrate(
                thrust_distribution,
                blade.root_cutout,
                blade.radius,
                collective=collective_pitch,
                blade=blade
            )

        return total_thrust

    def compute_hover_torque(self, collective_angle, air_density, induced_vel, **kwargs):
        """
        Calculate rotor torque in hover conditions using enhanced blade element theory.
        
        Args:
            collective_angle: Blade collective pitch angle
            air_density: Local air density
            induced_vel: Induced velocity at the rotor
            **kwargs: Additional parameters
        
        Returns:
            float: Total rotor torque in hover
        """
        total_torque = 0
        
        def torque_element_contribution(self, radius, **params):
            """Calculate local torque contribution at a blade element."""
            # Get blade parameters
            rotor_blade = params['blade']
            pitch_angle = params['collective']
            
            # Calculate local blade characteristics
            local_solidity = ((self.blade_count * rotor_blade.calculate_chord(radius)) / 
                            (np.pi * rotor_blade.radius))
            
            # Compute total blade pitch including twist
            effective_pitch = pitch_angle + rotor_blade.calculate_twist(radius)
            
            # Calculate inflow and aerodynamic angles
            inflow_ratio = self.compute_hover_inflow(
                radius, 
                effective_pitch, 
                local_solidity, 
                rotor_blade, 
                induced_vel
            )
            
            # Flow angle calculation
            inflow_angle = np.arctan(inflow_ratio * rotor_blade.radius / radius)
            
            # Get aerodynamic coefficients
            lift_coefficient = rotor_blade.calculate_lift_coefficient(effective_pitch - inflow_angle)
            drag_coefficient = rotor_blade.calculate_drag_coefficient(effective_pitch - inflow_angle)
            
            # Local velocity calculations
            tangential_velocity = self.angular_velocity * radius
            vertical_velocity = inflow_ratio * self.angular_velocity * rotor_blade.radius
            
            # Calculate local chord
            local_chord = rotor_blade.calculate_chord(radius)
            
            # Compute torque contribution with enhanced model
            return (radius * 0.5 * air_density * 
                    (tangential_velocity**2 + vertical_velocity**2) * 
                    local_chord * 
                    (drag_coefficient * np.cos(inflow_angle) + 
                    lift_coefficient * np.sin(inflow_angle)))
        
        # Sum torque contributions from all blades
        for blade in self.blade_array:
            total_torque += self.numerical_integrate(
                torque_element_contribution,
                blade.root_cutout,
                blade.radius,
                collective=collective_angle,
                blade=blade
            )

        return total_torque

    def compute_hover_inflow(self, radius, pitch, solidity, blade, induced_velocity):
        """Calculate inflow ratio in hover conditions using an enhanced model."""
        # Calculate advance ratio
        advance_ratio = induced_velocity / (self.angular_velocity * blade.radius)
        
        # Initialize inflow variables
        inflow = np.ones_like(pitch) * 0.01
        error = np.ones_like(pitch) * 1e10
        
        # Iterative solution for inflow
        while (abs(error) > 1e-8).all():
            previous_inflow = inflow.copy()
            
            # Calculate Prandtl's tip loss factor
            tip_loss_factor = (self.blade_count / 2) * ((1 - radius/blade.radius) / inflow)
            prandtl_factor = (2/np.pi) * np.arccos(np.exp(-tip_loss_factor))
            
            # Update inflow using enhanced model
            inflow = np.sqrt(
                ((solidity * blade.lift_coefficient_slope / (16 * prandtl_factor)) - 
                 (advance_ratio/2))**2 + 
                ((solidity * blade.lift_coefficient_slope / (8 * prandtl_factor)) * 
                 pitch * radius/blade.radius)
            ) - ((solidity * blade.lift_coefficient_slope / (16 * prandtl_factor)) - 
                 (advance_ratio/2))
            
            error = abs(inflow - previous_inflow)
            
        return inflow

    def get_average_chord(self):
        """Calculate the average chord length of the rotor blade."""
        def chord_distribution(self, radius, **kwargs):
            return self.blade_array[0].calculate_chord(radius)
            
        return self.numerical_integrate(
            chord_distribution, 
            self.blade_array[0].root_cutout, 
            self.blade_array[0].radius
        ) / 2

