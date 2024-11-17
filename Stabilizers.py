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
    