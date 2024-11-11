import numpy as np


# Class to handle data generation
class SimulationData:
    def __init__(self):
        pass

    # Functions to generate forces data
    def generate_forces_x(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + collective_pitch) + lateral_pitch + 0.5 * longitudinal_pitch
        return x, y

    def generate_forces_y(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + tail_rotor_collective) + longitudinal_pitch - 0.3 * lateral_pitch
        return x, y

    def generate_forces_z(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.cos(tail_rotor_collective + lateral_pitch) + 0.2 * longitudinal_pitch
        return x, y

    def generate_forces_xyz(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        # Separate the components
        force_x = np.sin(x + collective_pitch)
        force_y = np.cos(x + tail_rotor_collective)
        force_z = np.sin(x) * np.cos(tail_rotor_collective + lateral_pitch) + 0.2 * longitudinal_pitch

        # Combine the individual components
        return x, force_x, force_y, force_z

    # Functions to generate moments data
    def generate_moments_x(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + collective_pitch) + np.sin(
            x) + 0.4 * lateral_pitch - 0.2 * longitudinal_pitch + tail_rotor_collective
        return x, y

    def generate_moments_y(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + collective_pitch - 0.7 * lateral_pitch + 0.5 * longitudinal_pitch
        return x, y

    def generate_moments_z(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        y = np.cos(x + longitudinal_pitch) + collective_pitch - 0.6 * tail_rotor_collective + lateral_pitch
        return x, y

    def generate_moments_xyz(self, collective_pitch, lateral_pitch, longitudinal_pitch, tail_rotor_collective):
        x = np.linspace(0, 10, 100)
        # Separate the components
        moment_x = np.cos(x + collective_pitch) + np.sin(x)
        moment_y = np.sin(x) + collective_pitch - 0.7 * lateral_pitch + 0.5 * longitudinal_pitch
        moment_z = np.cos(x + longitudinal_pitch) + collective_pitch - 0.6 * tail_rotor_collective + lateral_pitch

        # Combine the individual components
        return x, moment_x, moment_y, moment_z
