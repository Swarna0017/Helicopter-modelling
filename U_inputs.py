# Plug all the input values here
# Please input the values in SI units or else you may need to convert them as you add
def Input_Plugger():
    Altitude        = 2000                      # Geometric altitude in metres
    MRR             = 2000                      # Main Rotor Radius in metres
    TRR             = 12                        # Tail Rotor Radius in meters
    V_inf           = 100                       # Flight velocity in m/s
    VW              = 50                        # Weight of the vehicle in kg
    MR_nb           = 3                         # Number of main rotor blades
    TR_nb           = 2                         # Number of Tail Rotor blades
    MR_Taper_ratio  = 0.8                       # Enter the taper ratio of the main rotor blades
    TR_Taper_ratio  = 0.8                       # Taper ratio of tail rotor blades
    MR_rc           = 0.78                      # Enter the root cut-out of main rotor blade
    TR_rc           = 0.1                       # Enter the root cut-out of tail rotor blade
    MR_root_twist   = 4                         # Enter the twist of main rotor blade root
    MR_tip_twist    = 0                         # Enter the twist of main rotor blade tip
    TR_root_twist   = 4                         # Enter the twist of tail rotor blade root
    TR_tip_twist    = 0                         # Enter the twist of tail rotor blade tip
    MR_chord        = 0.07                      # Main rotor blade chord
    TR_chord        = 0.09                      # Tail rotor blade chord
    HS_chord        = 0.09                      # Horizontal stabilizer blade chord
    MR_omega        = 123                       # Enter the RPM of the main rotor blades

    return Altitude, MRR, TRR, V_inf, VW, MR_nb, TR_nb, MR_Taper_ratio, TR_Taper_ratio, MR_rc, TR_rc, MR_root_twist, MR_tip_twist, TR_root_twist, TR_tip_twist, MR_chord, TR_chord, HS_chord, MR_omega
 
# Calling the Input_Plugger function to store the values in the following variables
Altitude, MRR, TRR, V_inf, VW, MR_nb, TR_nb, MR_Taper_ratio, TR_Taper_ratio, MR_rc, TR_rc, MR_root_twist, MR_tip_twist, TR_root_twist, TR_tip_twist, MR_chord, TR_chord, HS_chord, MR_omega = Input_Plugger()

# Creating an instance of these variables to use in other files and classes in the rest of the simulator flow.
class U_Inputs_Simulator:       # This is the class for the input variables of the flight simulator
    def __init__(self, Altitude: float, MRR: float, TRR: float, V_inf: float, VW: float, 
                 MR_nb: int, TR_nb: int, MR_Taper_ratio: float, TR_Taper_ratio: float, 
                 MR_rc: float, TR_rc: float, MR_root_twist: float, MR_tip_twist: float, 
                 TR_root_twist: float, TR_tip_twist: float, MR_chord: float, TR_chord: float, 
                 HS_chord: float):
        
        self.Altitude = Altitude
        self.MRR = MRR
        self.TRR = TRR
        self.V_inf = V_inf
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


class U_Inputs_Planner:
    def __init__(self, VW, ):
        self.VW=VW

# Flight_Simulator_Inputs=U_Inputs_Simulator()