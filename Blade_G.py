# This file takes in the bladed geometry parameters and returns chord length and pitch angle 
# of blade section at a given blade location
from Airfoil import Airfoil



class Blade:
    def __init__(self, chord: float, pitch: float, Airfoil: Airfoil, inertia: float):
        self.chord      = chord
        self.pitch      = pitch
        self.Airfoil    = Airfoil
        self.inertia    = inertia

    def get_cl(self, aoa: float):
        return self.airfoil.get_cl(aoa)
    
    def get_cd(self, aoa: float):
        return self.airfoil.get_cd(aoa)
    
    def calc_inst_forces(self):
        #calculates forces on the blade section
        pass
