# This file takes in the bladed geometry parameters and returns chord length and pitch angle 
# of blade section at a given blade location
# Inputs:: Bladed geometry parameters (radius, twist, taper)
# Outputs:: Chord length, pitch angle
from Airfoil import Airfoil
from U_inputs import U_Inputs_Simulator, Pilot_Inputs



class Blade:
    def __init__(self, simulator_inputs: U_Inputs_Simulator, Pilot_Inputs: Pilot_Inputs):
        self.MRR            = simulator_inputs.MRR
        self.TRR            = simulator_inputs.TRR
        self.MR_rc          = simulator_inputs.MR_rc
        self.MR_rb          = simulator_inputs.MR_nb
        self.MR_omega       = simulator_inputs.MR_omega
        self.MR_root_twist  = simulator_inputs.MR_root_twist
        self.MR_tip_twist   = simulator_inputs.MR_tip_twist
        self.MR_taper_ratio = simulator_inputs.MR_Taper_ratio
        self.MR_chord       = simulator_inputs.MR_chord
        self.TR_rc          = simulator_inputs.TR_rc
        self.TR_rb          = simulator_inputs.TR_nb
        self.TR_omega       = simulator_inputs.TR_omega
        self.TR_root_twist  = simulator_inputs.TR_root_twist
        self.TR_tip_twist   = simulator_inputs.TR_tip_twist
        self.TR_taper_ratio = simulator_inputs.TR_Taper_ratio
        self.TR_chord       = simulator_inputs.TR_chord
        self.MR_theta       = Pilot_Inputs.theta_0
        self.TR_theta       = Pilot_Inputs.theta_tail
        self.radial_sec     = self.Blade_sections(100)
        self.pitch_values   = self.Pitch()
        r_theta_values      = self.Blade_Pitch_dict()
        r_chord_values      = self.Blade_Chord_dict()



    def Blade_sections(self, no_of_sections=100):
        radial_sec=[self.MR_rc+i*(self.MRR-self.MR_rc)/(no_of_sections-1) for i in range (no_of_sections)]
        return radial_sec

    def Pitch(self):                                # Generates values of theta based on twist for each r
        F= (self.MR_root_twist-self.MR_tip_twist)/self.MRR*self.radial_sec
        theta_r=self.theta_0+F*self.radial_sec
        return theta_r
    
    def Blade_Pitch_dict(self):                     # Creates a dictionary corresponding to blade sections (r) and corresponding pitch (theta)
        return {r:theta for r,theta in zip(self.radial_sec, self.pitch_values)}

    def chord(self):                                # gives you the chord of the blade at a particular "R"
        chord_r = []
        for r in self.radial_sec:
            if r<self.MR_rc:
                chord_r.append(self.MR_rc)
            elif r>=self.MRR:
                chord_r.append(self.MRR*self.MR_taper_ratio)
            else:
                chord_r.append(self.MR_chord*(1-(r-self.MR_rc)/(self.MRR-self.MR_rc)*(1-self.MR_taper_ratio)))
            
        return chord_r
    
    def Blade_Chord_dict(self):
        return {r: chord_r for r, chord_r in zip(self.radial_sec, self.chord)}


