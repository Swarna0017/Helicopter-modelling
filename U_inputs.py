# Plug all the input values here
# Please input the values in SI units or else you may need to convert them as you add
import numpy as np

def Input_Plugger():
    Altitude        = 2000                      # Geometric altitude in metres
    MRR             = 0.762                         # Main Rotor Radius in metres
    TRR             = 0.5                       # Tail Rotor Radius in meters
    V               = 100                       # Flight (climb) velocity in m/s
    VW              = 50                        # Weight of the vehicle in kg
    MR_nb           = 3                         # Number of main rotor blades
    TR_nb           = 2                         # Number of Tail Rotor blades
    MR_Taper_ratio  = 0.8                       # Enter the taper ratio of the main rotor blades
    TR_Taper_ratio  = 0.8                       # Taper ratio of tail rotor blades
    MR_rc           = 0.125                      # Enter the root cut-out of main rotor blade
    TR_rc           = 0.1                       # Enter the root cut-out of tail rotor blade
    MR_root_twist   = 4                         # Enter the twist of main rotor blade root
    MR_tip_twist    = 0                         # Enter the twist of main rotor blade tip
    TR_root_twist   = 4                         # Enter the twist of tail rotor blade root
    TR_tip_twist    = 0                         # Enter the twist of tail rotor blade tip
    MR_chord        = 0.0508                      # Main rotor blade chord
    TR_chord        = 0.2                     # Tail rotor blade chord
    HS_chord        = 0.09                      # Horizontal stabilizer blade chord
    MR_omega        = 123                       # Enter the RPM of the main rotor blades
    MRA             = np.pi*MRR**2
    Iterations      = 100                       # the number of iterations for force calculation
    Cd_body         = 0.3                       # Enter the body drag coefficient for your helicopter, if unknown, use the following reference: 
    body_area       = 2  # (fuselage body area) # Cd = 0.2 to 0.4 for small helicopters, 0.3 to 0.5 for medium helicpters, 0.4 to 0.6 for militray helicopters, and 0.15 to 0.25 for highly streamlined models


    return Altitude, MRR, TRR, V, VW, MR_nb, TR_nb, MR_Taper_ratio, TR_Taper_ratio, MR_rc, TR_rc, MR_root_twist, MR_tip_twist, TR_root_twist, TR_tip_twist, MR_chord, TR_chord, HS_chord, MR_omega, MRA, Iterations, Cd_body, body_area 

def Pilot_Input_Plugger():
    theta_0         = 10                         # Main Rotor Collective pitch 
    theta_1s        = 0                          # Main Rotor longitudinal Cyclic
    theta_1c        = 0                          # Main Rotor lateral Cyclic
    theta_tail      = 0                          # Tail Rotor Collective
    return theta_0, theta_1s, theta_1c, theta_tail
 
# Calling the Input_Plugger function to store the values in the following variables
Altitude, MRR, TRR, V, VW, MR_nb, TR_nb, MR_Taper_ratio, TR_Taper_ratio, MR_rc, TR_rc, MR_root_twist, MR_tip_twist, TR_root_twist, TR_tip_twist, MR_chord, TR_chord, HS_chord, MR_omega, MRA, Iterations, Cd_body, body_area = Input_Plugger()
theta_0, theta_1s, theta_1c, theta_tail = Pilot_Input_Plugger()

# Creating an instance of these variables to use in other files and classes in the rest of the simulator flow.

class U_Inputs_Simulator:                                                                   # This is the class for the input variables of the flight simulator
    def __init__(self, Altitude: float, MRR: float, TRR: float, V: float, VW: float, 
                 MR_nb: int, TR_nb: int, MR_Taper_ratio: float, TR_Taper_ratio: float, 
                 MR_rc: float, TR_rc: float, MR_root_twist: float, MR_tip_twist: float, 
                 TR_root_twist: float, TR_tip_twist: float, MR_chord: float, TR_chord: float, 
                 HS_chord: float, MR_omega: float, MRA: float, Iterations, Cd_body: float, body_area:float):
        
        self.Altitude = Altitude
        self.MRR = MRR
        self.TRR = TRR
        self.V = V
        self.VW = VW
        self.MR_nb = MR_nb
        self.TR_nb = TR_nb
        self.MR_Taper_ratio = MR_Taper_ratio
        self.TR_Taper_ratio = TR_Taper_ratio
        self.MR_rc = MR_rc
        self.TR_rc = TR_rc
        self.MR_root_twist = MR_root_twist
        self.MR_tip_twist = MR_tip_twist
        self.TR_root_twist = TR_root_twist
        self.TR_tip_twist = TR_tip_twist
        self.MR_chord = MR_chord
        self.TR_chord = TR_chord
        self.HS_chord = HS_chord
        self.MR_omega = MR_omega
        self.MRA      = MRA
        self.Iterations =Iterations
        self.Cd_body    =Cd_body
        self.body_area  =body_area

class Pilot_Inputs():                                                                       # This is the class for pilot inputs of the flight simulator
    def __init__(self, theta_0:float, theta_1s:float, theta_1c:float, theta_tail):
        self.theta_0 = theta_0
        self.theta_1s = theta_1s
        self.theta_1c = theta_1c
        self.theta_tail = theta_tail

class U_Inputs_Planner:
    def __init__(self, VW, ):
        self.VW=VW

# Flight_Simulator_Inputs=U_Inputs_Simulator()