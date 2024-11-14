A. How to Run
    1. Necessary libraries:
    # Please make sure the following libraries are downloaded before you run the code on your own system.
    a. matplotlib.pyplot
        Install using the command: pip install matplotlib
    b. numpy
        Install using the command: pip install numpy
    2. Running the code
    The code flow is simple and modular. The following are the steps to run:
    a. Enter all your inputs in U_inputs file (Please make sure they are entered in SI units)
    b. Open and Run the Flight_Simulator.py file (The GUI has been integrated in the same file)

B. Code flow
    Different files of the code serve different functions. The function/need of each of the files are given for reference:
    1. AirData.py:
        This file houses the class (Atmosphere) for calculating environmental data at different altitudes 
        viz. speed of sound, Density, Temperature, Pressure, etc.
    2. U_inputs.py:
        This is the file where all the user inputs are to be plugged in. 
        Two classes (for Flight Simulator and Mission Planner inputs have been created for convenience)
    3. Airfoil.py:
        Gives you the relevant data required for your Airfoil (Cl, Cm, Cd, etc.) at required AOA.
        Databases of different Airfoils have been stored separately for the purpose
    4. Flight_Simulator.py:
        This file brings together all the files and classes to give you your required output
    5. Fuselage.py:
        All the relevant calculations for the Fuselage
    6. Blade.py:
        Does all the necessary blade-related computations
    7. Mission_Planner.py:
        Brings all the classes together to serve the purpose of the mission planner.
    8. Airfoil databases (NACA files):
        These files host the required relevant information of/for the airfoils at particular angles of attack.
    9. Stabilizers.py
        To compute forces on the horizontal and vertical stabilizers based on stabilizer geometry and 
        placement.
    10. Vehicle_Dynamics.py:
        Takes in instantaneous forces and moments from all components of the 
        vehicle (main rotor, tail rotor, stabilizers, etc.) and computes the net instantaneous forces 
        and moments (all three axes) about the vehicle centric reference frame, accounting for the 
        placement of the components
    11. Cyclic_Integrator.py:
        Integrates forces and moments about one complete rotation (azimuth circle) for 
        performance computations (thrust, power, moments, etc.)
    12. Instantaneous_Integrator.py:
        Integrates instantaneous forces and moments on individual 
        blades. It relies on all of the above functions in addition to the knowledge of instantaneous 
        blade location to give answers.

C. Variables used:
    # Following is the list of all the major variables used throughout the code (for clarity):
    # Tip: All main rotor variables begin with MR, all tail rotor variables, similarly, begin with TR
    1.	Altitude : Flight altitude of the helicopter
    2.	MRR: Main Rotor radius
    3.	TRR: Tail Rotor radius
    4.	V: Climb velocity
    5.	VW: Vehicle Weight
    6.	MR_nb: Main Rotor number of blades
    7.	TR_nb: Tail rotor number of blades
    8.	MR_Taper_ratio: Main Rotor Taper ratio
    9.	TR_Taper_ratio: Tail Rotor Taper ratio
    10.	MR_rc: Main Rotor Root cut-out
    11.	TR_rc: Tail Rotor Root cut-out
    12.	MR_root_twist: Main rotor Root twist
    13.	MR_tip_twist: Main rotor tip twist
    14.	TR_root_twist: Tail Rotor root twist
    15.	TR_tip_twist: Tail rotor tip twist
    16.	MR_chord: Main rotor blade chord
    17.	TR_chord: Tail rotor blade chord
    18.	HS_chord: Horizontal stabilizer blade chord
    19. MR_omega: RPM of the main rotor blades in user inputs
    10. TR_omega: RPM of the main rotor blades in user inputs
    11. Cd_body: Coefficient of drag for the helicopter
    12. body_area: Effective area of the helicopter contributing to drag
    13. alpha_tpp: Angle made my tip path plane with the relative wind
    14. psi: azimuthal angle 
    15. v/vi: induced velocity
