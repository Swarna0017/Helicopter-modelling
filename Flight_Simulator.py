# The GUI appends here 

import tkinter as tk 
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

from U_inputs import Input_Plugger, Pilot_Input_Plugger

# Define the function to gather user inputs
def get_user_inputs():
    user_inputs = {
        'Altitude'        : float(altitude_Entry.get()),
        'MRR'             : float(main_radius_Entry.get()),
        'TRR'             : float(tail_radius_Entry.get()),
        'V'               : float(velocity_Entry.get()),
        'VW'              : float(weight_Entry.get()),
        'MR_nb'           : int(main_nublades_Entry.get()),
        'TR_nb'           : int(tail_nublades_Entry.get()),
        'MR_Taper_ratio'  : float(main_taper_Entry.get()),
        'TR_Taper_ratio'  : float(tail_taper_Entry.get()),
        'MR_rc'           : float(main_rootcut_Entry.get()),
        'TR_rc'           : float(tail_rootcut_Entry.get()),
        'MR_root_twist'   : float(main_root_twist_Entry.get()),
        'MR_tip_twist'    : float(main_tip_twist_Entry.get()),
        'TR_root_twist'   : float(tail_root_twist_Entry.get()),
        'TR_tip_twist'    : float(tail_tip_twist_Entry.get()),
        'MR_chord'        : float(main_chord_Entry.get()),
        'TR_chord'        : float(tail_chord_Entry.get()),
        'HS_chord'        : float(horizontal_chord_Entry.get()),
        'MR_omega'        : float(main_omega_Entry.get()),
        'MRA'             : float(body_area_Entry.get()),
        'Iterations'      : int(iterations_Entry.get()),
        'Cd_body'         : float(drag_coefficient_Entry.get()),
        'body_area'       : float(body_area_Entry.get())
    }
    print(user_inputs)  # Debugging line
    return user_inputs

def get_pilot_inputs():
    pilot_inputs = {
        'theta_0'         : float(collective_Entry.get()),
        'theta_1s'        : float(cyclic_longitudinal_Entry.get()),
        'theta_1c'        : float(cyclic_lateral_Entry.get()),
        'theta_tail'      : float(tail_collective_Entry.get())
    }
    print(pilot_inputs)  # Debugging line
    return pilot_inputs

# Main window
root = tk.Tk()
root.title("Helicopter Flight Simulator")

image = Image.open("bgimg.jpeg")
image_width, image_height = image.size
root.geometry(f"{image_width}x{image_height}")
photo = ImageTk.PhotoImage(image)
root.photo = photo  
canvas = tk.Canvas(root, width=image_width, height=image_height)
canvas.grid(row=0, column=0, rowspan=100, columnspan=20)  
canvas.create_image(0, 0, image=photo, anchor='nw')


tk.Label(root, text="FLIGHT SIMULATOR", font=("Times New Roman", 18, "bold")).grid(row=1, column=0, columnspan=2, pady=10)

tk.Label(root, text="A. Main Rotor Inputs:", font=("Arial", 14, "bold")).grid(row=2, column=0, columnspan=2, pady=10)

tk.Label(root, text="Main Rotor Radius").grid(row=6, column=0)
main_radius_Entry = ttk.Entry(root)
main_radius_Entry.grid(row=6, column=1)

tk.Label(root, text="Main Rotor Number of Blades").grid(row=7, column=0)
main_nublades_Entry = ttk.Entry(root)
main_nublades_Entry.grid(row=7, column=1)

tk.Label(root, text="Main Rotor Taper Ratio").grid(row=8, column=0)
main_taper_Entry = ttk.Entry(root)
main_taper_Entry.grid(row=8, column=1)

tk.Label(root, text="Main Rotor Root Cutout").grid(row=9, column=0)
main_rootcut_Entry = ttk.Entry(root)
main_rootcut_Entry.grid(row=9, column=1)

tk.Label(root, text="Main Rotor Twist (Root)").grid(row=10, column=0)
main_root_twist_Entry = ttk.Entry(root)
main_root_twist_Entry.grid(row=10, column=1)

tk.Label(root, text="Main Rotor Twist (Tip)").grid(row=11, column=0)
main_tip_twist_Entry = ttk.Entry(root)
main_tip_twist_Entry.grid(row=11, column=1)

tk.Label(root, text="Main Rotor Chord").grid(row=12, column=0)
main_chord_Entry = ttk.Entry(root)
main_chord_Entry.grid(row=12, column=1)

tk.Label(root, text="Main Rotor RPM").grid(row=13, column=0)
main_omega_Entry = ttk.Entry(root)
main_omega_Entry.grid(row=13, column=1)

# Tail rotor I/P parameters
tk.Label(root, text="B. Tail Rotor Inputs:", font=("Arial", 14, "bold")).grid(row=15, column=0, columnspan=2, pady=10)

tk.Label(root, text="Tail Rotor Radius").grid(row=16, column=0)
tail_radius_Entry = ttk.Entry(root)
tail_radius_Entry.grid(row=16, column=1)

tk.Label(root, text="Tail Rotor Number of Blades").grid(row=17, column=0)
tail_nublades_Entry = ttk.Entry(root)
tail_nublades_Entry.grid(row=17, column=1)

tk.Label(root, text="Tail Rotor Taper Ratio").grid(row=18, column=0)
tail_taper_Entry = ttk.Entry(root)
tail_taper_Entry.grid(row=18, column=1)

tk.Label(root, text="Tail Rotor Root Cutout").grid(row=19, column=0)
tail_rootcut_Entry = ttk.Entry(root)
tail_rootcut_Entry.grid(row=19, column=1)

tk.Label(root, text="Tail Rotor Twist (Root)").grid(row=20, column=0)
tail_root_twist_Entry = ttk.Entry(root)
tail_root_twist_Entry.grid(row=20, column=1)

tk.Label(root, text="Tail Rotor Twist (Tip)").grid(row=21, column=0)
tail_tip_twist_Entry = ttk.Entry(root)
tail_tip_twist_Entry.grid(row=21, column=1)

tk.Label(root, text="Tail Rotor Chord").grid(row=22, column=0)
tail_chord_Entry = ttk.Entry(root)
tail_chord_Entry.grid(row=22, column=1)

# Miscellaneous section for other inputs
tk.Label(root, text="C. Miscellaneous Inputs:", font=("Arial", 14, "bold")).grid(row=8, column=6, columnspan=2, pady=10)

tk.Label(root, text="Altitude").grid(row=9, column=6)
altitude_Entry = ttk.Entry(root)
altitude_Entry.grid(row=9, column=7)

tk.Label(root, text="Flight Velocity").grid(row=10, column=6)
velocity_Entry = ttk.Entry(root)
velocity_Entry.grid(row=10, column=7)

tk.Label(root, text="Vehicle Weight").grid(row=11, column=6)
weight_Entry = ttk.Entry(root)
weight_Entry.grid(row=11, column=7)

tk.Label(root, text="Iterations").grid(row=12, column=6)
iterations_Entry = ttk.Entry(root)
iterations_Entry.grid(row=12, column=7)

tk.Label(root, text="Drag Coefficient").grid(row=13, column=6)
drag_coefficient_Entry = ttk.Entry(root)
drag_coefficient_Entry.grid(row=13, column=7)

tk.Label(root, text="Body Area").grid(row=14, column=6)
body_area_Entry = ttk.Entry(root)
body_area_Entry.grid(row=14, column=7)

tk.Label(root, text="Horizontal Stabilizer Chord").grid(row=15, column=6)
horizontal_chord_Entry = ttk.Entry(root)
horizontal_chord_Entry.grid(row=15, column=7)

# Pilot controls section
tk.Label(root, text="D. Pilot Inputs:", font=("Arial", 16, "bold")).grid(row=2, column=6, columnspan=2, pady=10)

tk.Label(root, text="Main Rotor Collective").grid(row=3, column=6)
collective_Entry = ttk.Entry(root)
collective_Entry.grid(row=3, column=7)

tk.Label(root, text="Longitudinal Cyclic (Main Rotor)").grid(row=4, column=6)
cyclic_longitudinal_Entry = ttk.Entry(root)
cyclic_longitudinal_Entry.grid(row=4, column=7)

tk.Label(root, text="Lateral Cyclic (Main Rotor)").grid(row=5, column=6)
cyclic_lateral_Entry = ttk.Entry(root)
cyclic_lateral_Entry.grid(row=5, column=7)

tk.Label(root, text="Tail Rotor Collective").grid(row=6, column=6)
tail_collective_Entry = ttk.Entry(root)
tail_collective_Entry.grid(row=6, column=7)

# Submit Button to get inputs
def on_submit():
    user_inputs = get_user_inputs()
    pilot_inputs = get_pilot_inputs()
    # Here you can pass these inputs to the simulator or store them for later use
    print("User Inputs:", user_inputs)
    print("Pilot Inputs:", pilot_inputs)

submit_button = ttk.Button(root, text="Submit", command=lambda:on_submit)
submit_button.grid(row=16, column=6, columnspan=8, pady=10)

root.mainloop()

