# Calculates inflow ratios for different flight regimes
from U_inputs import U_Inputs_Simulator, Pilot_Inputs
from Blade_G import Blade
from AirData import Atmosphere
from scipy.optimize import fsolve
import math
import numpy as np

class v_calculator:
    def __init__(self, Blade: Blade, simulator_inputs: U_Inputs_Simulator, blade: Blade, pilot_inputs: Pilot_Inputs, atmosphere: Atmosphere):
        self.V      = simulator_inputs.V
        self.omega  = simulator_inputs.MR_omega*2/60*math.pi
        self.VW     = simulator_inputs.VW
        self.solidity = blade.Disk_solidity()
        self.MR_omega = simulator_inputs.MR_omega
        self.MRR      = simulator_inputs.MRR
        self.MRA      = simulator_inputs.MRA
        self.theta    = pilot_inputs.theta_0
        self.a0       = 5.75
        self.rho      = atmosphere.rho_calc()

    def v_hover(self):
        v = -self.V*0.5 + math.sqrt((self.V*0.5)**2+(self.VW/(2*self.rho*self.A)))                 # Induced velocity for hover, Momentum theory
        return v
    
    def v_forward(self, alpha_tpp, T):
        def v_calc(v):
            return 2*self.rho*self.MRA*v*np.sqrt(self.V**2*np.cos(alpha_tpp)*np.cos(alpha_tpp) + (self.V*np.sin(alpha_tpp)+v)*(self.V*np.sin(alpha_tpp)+v))-T
        vii=fsolve(v_calc, self.v_hover())
        return vii

    
    def Prandtl_tip_loss_implemeter(self,r):
        f =  0.5*self.MR_nb*(self.MRR-r)*self.omega/(self.V+self.v)
        if(f>500):
            F=1
        else:
            F=2*np.arccos(np.exp(-f))/np.pi
        return F
    
    def lambda_vertical(self,r):
        F =  self.Prandtl_tip_loss_implemeter(r)
        lambda_c = self.V/(self.omega*self.R)
        k = ((self.solidity*self.a0/(16*F))-(lambda_c/2))
        lambda_i =  np.sqrt(k*k+(self.solidity*self.a0*self.theta*r)/(8*self.MRR*F))-k
        lambda_i = lambda_i - self.V/(self.omega*self.MRR)
        return lambda_i
    

    def lambda_forward(self, psi, r, alpha_tpp, T):
        vi = self.v_forward(alpha_tpp, T)
        lambda_g = (vi + self.V*np.sin(alpha_tpp))/(self.MRR*self.omega)
        lambda_ig = (vi)/(self.MRR*self.omega)
        mu = self.V*np.cos(alpha_tpp)/(self.omega*self.MRR)
        lambda_i = lambda_ig*(1+(((4/3)*mu/lambda_g)/(1.2+mu/lambda_g))*(r/self.MRR)*np.cos(psi))
        return lambda_i
        