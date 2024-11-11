# Plug all the input values here
# Please input the values in SI units or else you may need to convert them as you add
def Input_Plugger():
    Altitude= 2000              # Geometric altitude in metres
    MRR= 2000                   # Main Rotor Radius in metres
    TRR= 12                     # Tail Rotor Radius in meters
    V_inf= 100                  # Flight velocity in m/s
    VW=50                       # Weight of the vehicle in kg

    return Altitude, MRR, TRR

class U_Inputs_Simulator:
    def __init__(self, Altitude: float, MRR: float, TRR: float):
        self.Altitude=Altitude
        self.MRR=MRR

class U_Inputs_Simulator:
    def __init__(self, VW, ):
        self.VW=VW