# This file computes the forces and moments on individual blades.
from U_inputs import U_Inputs_Simulator, Pilot_Inputs                         # Specified input files are called from here
from AirData import Atmosphere                                  # Fetches the required environmental sdata
from Blade_G import Blade 
from Airfoil import Airfoil_data                                      # For importing the relevant blade parameters like chord length, taper, etc.
from Inflow import v_calculator
import math
import numpy as np

# All classes for implementing different hover performance prediction methods here
class BEMT_Implementer():
    def __init__(self, simulator_inputs=U_Inputs_Simulator, pilot_inputs=Pilot_Inputs, Blade=Blade, Atmosphere_data= Atmosphere, Airfoil=Airfoil_data, v_data= v_calculator):
        self.rho_0              = 1.225             # kg/m^3   
        Atmosphere_data         = Atmosphere(simulator_inputs, pilot_inputs)
        Blade                   = Blade(simulator_inputs=simulator_inputs, pilot_inputs=pilot_inputs)     
        self.simulator_inputs   = simulator_inputs
        self.pilot_inputs       = Pilot_Inputs 
        self.rho                = Atmosphere_data.rho_calc()
        self.V                  = simulator_inputs.V
        self.MRR                = simulator_inputs.MRR
        self.MRA                = simulator_inputs.MRA
        self.omega              = simulator_inputs.MR_omega
        self.MR_nb              = simulator_inputs.MR_nb
        self.MR_rc              = simulator_inputs.MR_rc
        self.theta              = pilot_inputs.theta_0
        self.MR_omega           = (simulator_inputs.MR_omega)*math.pi*2/60
        self.VW                 = simulator_inputs.VW
        self.A                  = simulator_inputs.MRA
        self.r                  = Blade.Blade_sections(10)
        self.v                  = v_data.v_hover(self)  # Accessing v_hover from the instance
        self.chord_r            = Blade.chord()
        self.phi                = Airfoil.Phi(self)
        self.range              = simulator_inputs.Iterations
        self.v_data             = v_data(Blade, simulator_inputs)  # Creating an instance of v_calculator
        self.aoa                = Airfoil.AOA(self)  
        self.dr                 = Blade.dr
        self.MR_chord           = simulator_inputs.MR_chord


        self.Thrust, self.Torque, self.Power = self.BEMT_Solver()
        self.Ct, self.Cq, self.Cp = self.Coeff_finder(self.Thrust, self.Torque, self.Power)
        print(f"density: {self.rho:.3f} kg/m^3")
        print(f"T={self.Thrust:.3f} N\nQ={self.Torque:.3f} Nm\nP={self.Power:.3f} W")
        print(f"Ct={self.Ct:.3f}\nCq={self.Cq:.3f}\nCp={self.Cp:.3f}")

    def Velocities(self,r):
        Ut = self.omega*r
        Up = self.V + self.V
        return Ut, Up
    
    def Prandtl_tip_loss_implemeter(self,r):
        f =  0.5*self.MR_nb*(self.MRR-r)*self.omega/(self.V+self.v)
        if(-f>500):
            F=1
        else:
            F=2*np.arccos(np.exp(-f))/np.pi
        return F
    
    def Coeff_finder(self, Thrust, Torque, Power):
        Ct = Thrust/(self.rho*self.MRA*(self.MR_omega*self.MRR)**2)
        Cq = Torque/(self.rho*self.MRR*self.MRA*(self.MR_omega*self.MRR)**2)
        Cp = Power/(self.rho*self.MRA*(self.MR_omega*self.MRR)**3)
        return Ct, Cq, Cp

    def BEMT_Solver(self):       # Defining a solver for calculating thrust, torque and power
        self.Thrust=0
        self.Torque=0 
        for i in range(len(self.r)):
            r               = float(self.r[i])
            chord           = float(self.chord_r[i])
            Ut, Up          = self.Velocities(r)
            phi             = float(self.phi[i])
            aoa             = float(self.aoa[i])
            cl, cd          = Airfoil_data.get_ClCd(self)
            F               = self.Prandtl_tip_loss_implemeter(r)
            dT              = 0.5*self.rho*(Ut**2+Up**2)*chord*(cl[i]*np.cos(phi)-cd[i]*np.sin(phi))*self.dr*F
            dQ              = 0.5*self.rho*(Ut**2+Up**2)*chord*(cl[i]*np.cos(phi)-cd[i]*np.sin(phi))*self.dr*F*r
            self.Thrust    += dT*self.MR_nb
            self.Torque    += dQ*self.MR_nb
        self.Power=self.Torque*self.MR_omega
        # print(f"Iteration {i}: r={r}, chord={chord}, Ut={Ut}, Up={Up}, phi={phi}, aoa={aoa}, cl={cl}, cd={cd}, F={F}, dT={dT}, dQ={dQ}")

        return self.Thrust, self.Torque, self.Power
    
class Forward_flight_analyzer():
    def __init__(self, simulator_inputs: U_Inputs_Simulator, pilot_inputs: Pilot_Inputs, blade:Blade, Atmosphere_data: Atmosphere, Airfoil=Airfoil_data, v_data= v_calculator):
        self.rho_0      = 1.225             #kg/m^3   
        Atmosphere_data = Atmosphere(simulator_inputs, rho_0=1.225, T_0=298, P_0=101325, Temp_grad=-6.5e-3)
        Blade = Blade(simulator_inputs=simulator_inputs, pilot_inputs=pilot_inputs)     
        self.simulator_inputs=simulator_inputs
        self.pilot_inputs=Pilot_Inputs 
        self.rho        = Atmosphere_data.rho_calc()
        self.V          = simulator_inputs.V
        self.MRR        = simulator_inputs.MRR
        self.MRA        = simulator_inputs.MRA
        self.omega      = simulator_inputs.MR_omega
        self.MR_nb      = simulator_inputs.MR_nb
        self.MR_rc      = simulator_inputs.MR_rc
        self.omega      = (simulator_inputs.MR_omega)*math.pi*2/60
        self.r          = Blade.Blade_sections(10)
        self.chord_r    = Blade.chord()
        self.VW         = simulator_inputs.VW
        self.A          = simulator_inputs.MRA
        self.phi        = Airfoil.Phi(self)
        self.range      = simulator_inputs.No_of_iterations
        self.v_data     = v_data(Blade, simulator_inputs)  # Creating an instance of v_calculator
        self.v          = self.v_data.v_hover()  # Accessing v_hover from the instance
        self.aoa        = Airfoil.AOA(self, self.phi)  
        self.dr         = Blade.dr
        self.MR_chord   = simulator_inputs.MR_chord
        self.Cd_body    = simulator_inputs.Cd_body
        self.body_area  = simulator_inputs.body_area
        self.theta_0    = pilot_inputs.theta_0
        self.theta_1c   = pilot_inputs.theta_1c
        self.theta_1s   = pilot_inputs.theta_1s
        self.beta_0     = 8
        self.Cd0        = 0.09

    def alpha_TPP_calc(self):
        Body_drag=self.Cd_body*0.5*self.rho*self.V*self.V*self.body_area
        alpha_tpp=np.arctan(Body_drag/self.VW)
        return alpha_tpp
    
    def Forward_Flight_calculations(self, psi, T, r):
        theta_root = self.theta_0 + self.theta_1s*np.sin(psi) + self.theta_1c*np.cos(psi)
        self.dr                 = Blade.dr
        theta = Blade.Pitch()
        #calculating induced velocity for the given climb speed and blade dimensions
        lambda_forward = v_calculator.lambda_forward(psi, r, self.alpha_tpp, T)
        v_i = lambda_forward*self.MRR*self.omega
        #calculating Perpendicular and Tangential Components
        U_P = self.V*np.sin(self.alpha_tpp) + v_i + self.V*np.cos(psi)*np.sin(self.beta_0)
        U_T = self.omega*r + self.V*np.cos(self.alpha_tpp)*np.sin(psi)
        cl, cd = Airfoil_data.get_ClCd()
        chord_r = Blade.chord()
        #calculating Angle of Attack at each section
        phi = np.arctan2(U_P, U_T)
        alpha_eff = theta - phi
        #arrays to store sectional Thrust and Drag
        T = []
        D = []
        #Assuming stall after angles <-15 and >15 degrees
        for i in range(np.size(alpha_eff)):
            self.aoa=alpha_eff
            if (alpha_eff[i]>-15*np.pi/180 and alpha_eff[i]<15*np.pi/180):
                cl, cd = Airfoil_data.get_ClCd()
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
            else:
                cl = 0
                cd = self.Cd0
                #calculating Thrust and Drag at a given section and appending it
                T.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cl*np.cos(phi[i])-cd*np.sin(phi[i]))*self.dr)
                D.append((self.rho*1/2)*(U_P[i]*U_P[i]+U_T[i]*U_T[i])*chord_r*(cd*np.cos(phi[i])+cl*np.sin(phi[i]))*self.dr)
        T = np.array(T)
        D = np.array(D)
        #returning the given sectional forces array                 
        return T, D
    
    
def MT_Implementer():
    def Thrust_Calculator(self):
        Thrust_hover=2*self.v**2*self.rho*self.A                    # This calculates the Thrust for hover case (Momentum Theory)
        Thrust_Climb=2*self.rho*self.A*(self.v+self.V)              # This calculates the Thrust in case of climb (Momentum Theory)
        return Thrust_hover, Thrust_Climb

    def Power_Calculator(self):
        Thrust_hover, _=self.Thrust_Calculator()
        Power=Thrust_hover*self.v
        return Power


    def Coefficient_Calculator(self):
        Thrust_hover, _=self.Thrust_Calculator()
        Power=self.Power_Calculator()
        Ct=Thrust_hover/(self.rho*self.A*(self.omega*self.MRR)**2)
        Cp=Power/(self.rho*self.A*(self.omega*self.MRR)**3)
        return Ct, Cp
    
    # def BET_Implementer():
    

    # def BEMT_Implementor():
      
