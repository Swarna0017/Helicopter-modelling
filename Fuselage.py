class Fuselage:
    def __init__(self, cd: float, area: float, position: tuple):
        self.cd=cd
        self.area=area
        self.position=position

    def calc_drag(self, velocity: float):
        # Computes drag force on the fuselage
        return 0.5*self.cd*self.area*velocity**2
    
    