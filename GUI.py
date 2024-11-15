import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
# testing commit
'''
User I/Ps (to be) taken (For my reference):

bld_main_radius
bld_main_rootchord
bld_main_nublades
bld_main_rootcut
bld_main_taper
bld_main_twist
bld_tail_radius
bld_tail_chord
bld_tail_nublades
bld_tail_rootcut
bld_tail_taper
bld_tail_twist

'''

def get_user_inputs():
    # takes User inputs from the Entry fields.
    user_inputs = {
        'main_radius'       : float(main_radius_Entry.get()),
        'main_number'       : int(main_number_Entry.get()),
        'main_omega'        : float(main_omega_Entry.get()),
        'main_root_radius'  : float(main_rc_Entry.get()),
        'main_taper'        : float(main_taper_Entry.get()),
        'main_twist'        : float(main_twist_Entry.get()),
        'tail_radius'       : float(tail_radius_Entry.get()),
        'tail_number'       : int(tail_number_Entry.get()),
        'tail_omega'        : float(tail_omega_Entry.get()),
        'tail_root_cutout'  : float(tail_root_cutout_Entry.get()),
        'tail_taper'        : float(tail_taper_Entry.get()),
        'tail_twist'        : float(tail_twist_Entry.get()),
    }
    print(user_inputs)  # For debugging
    return user_inputs


# Function to get pilot inputs
def get_pilot_inputs():
    pilot_inputs = {
    'main_collective'   : float(main_collective_Entry.get()),
    'main_cyclic_a1'    : float(main_cyclic_a1_Entry.get()),
    'main_cyclic_a2'    : float(main_cyclic_a2_Entry.get()),
    'tail_collective'   : float(tail_collective_Entry.get())
    }
    print(pilot_inputs)
    return pilot_inputs


# Main window
root = tk.Tk()
root.title("Helicopter Flight Simulator")

# For the bg image
image = Image.open("bgimg.jfif") 
photo = ImageTk.PhotoImage(image)

# This just puts the image in the background (You guys are welcomed to change this image if you find a better one :D )
canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
canvas.grid(row=0, column=0, rowspan=100, columnspan=20) 
canvas.create_image(0, 0, image=photo, anchor='nw')


# Input fields for the main rotor I/Ps
tk.Label(root, text="FLIGHT SIMULATOR", font=("Times New Roman", 18, "bold")).grid(row=1, column=0, columnspan=2, pady=10)

tk.Label(root, text="A. Main Rotor Inputs:", font=("Arial", 14, "bold")).grid(row=2, column=0, columnspan=2, pady=10)

tk.Label(root, text="Main Rotor Radius").grid(row=6, column=0)
main_radius_Entry = ttk.Entry(root)
main_radius_Entry.grid(row=6, column=1)

tk.Label(root, text="Main Rotor Frequency").grid(row=7, column=0)
main_number_Entry = ttk.Entry(root)
main_number_Entry.grid(row=7, column=1)

tk.Label(root, text="Main Rotor Omega").grid(row=8, column=0)
main_omega_Entry = ttk.Entry(root)
main_omega_Entry.grid(row=8, column=1)

tk.Label(root, text="Main Rotor Root Cutout").grid(row=9, column=0)
main_rc_Entry = ttk.Entry(root)
main_rc_Entry.grid(row=9, column=1)

tk.Label(root, text="Main Rotor Taper").grid(row=10, column=0)
main_taper_Entry = ttk.Entry(root)
main_taper_Entry.grid(row=10, column=1)

tk.Label(root, text="Main Rotor Twist").grid(row=11, column=0)
main_twist_Entry = ttk.Entry(root)
main_twist_Entry.grid(row=11, column=1)

# Tail rotor I/P parameters
        # for my reference      
        # 'TR_radius': float(tail_radius_Entry.get()),
        # 'TR_nu_blades': int(tail_number_Entry.get()),
        # 'TR_omega': float(tail_omega_Entry.get()),
        # 'TR_root_cutout': float(tail_root_cutout_Entry.get()),
        # 'TR_taper': float(tail_taper_Entry.get()),
        # 'TR_twist': float(tail_twist_Entry.get()),
tk.Label(root, text="B. Tail Rotor Inputs:", font=("Arial", 14, "bold")).grid(row=15, column=0, columnspan=2, pady=10)

tk.Label(root, text="Tail Rotor radius").grid(row=19, column=0)
tail_radius_Entry = ttk.Entry(root)
tail_radius_Entry.grid(row=19, column=1)

tk.Label(root, text="Tail Rotor Frequency").grid(row=20, column=0)
tail_number_Entry = ttk.Entry(root)
tail_number_Entry.grid(row=20, column=1)

tk.Label(root, text="Tail Rotor omega").grid(row=21, column=0)
tail_omega_Entry = ttk.Entry(root)
tail_omega_Entry.grid(row=21, column=1)

tk.Label(root, text="Tail Rotor Root Cutout").grid(row=22, column=0)
tail_root_cutout_Entry = ttk.Entry(root)
tail_root_cutout_Entry.grid(row=22, column=1)

tk.Label(root, text="Tail Rotor Taper").grid(row=23, column=0)
tail_taper_Entry=ttk.Entry(root)
tail_taper_Entry.grid(row=23, column=1)

tk.Label(root, text="Tail Rotor Twist").grid(row=24, column=0)
tail_twist_Entry=ttk.Entry(root)
tail_twist_Entry.grid(row=24, column=1)




# Input fields for the pilot controls
tk.Label(root, text="C. Pilot Inputs:", font=("Arial", 16, "bold")).grid(row=2, column=6, columnspan=2, pady=10)

tk.Label(root, text="Main Rotor Collective Pitch (Degrees)").grid(row=6, column=6)
main_collective_Entry = ttk.Entry(root)
main_collective_Entry.grid(row=6, column=7)

tk.Label(root, text="Main Rotor Cyclic Pitch (Degrees)").grid(row=7, column=6)
main_cyclic_a1_Entry = ttk.Entry(root)
main_cyclic_a1_Entry.grid(row=7, column=7)

tk.Label(root, text="Main Rotor Cyclic Roll (Degrees)").grid(row=8, column=6)
main_cyclic_a2_Entry = ttk.Entry(root)
main_cyclic_a2_Entry.grid(row=8, column=7)

tk.Label(root, text="Tail Rotor Collective Pitch (Degrees)").grid(row=9, column=6)
tail_collective_Entry = ttk.Entry(root)
tail_collective_Entry.grid(row=9, column=7)

# Button to submit the inputs
submit_button = ttk.Button(root, text="Get User Inputs", command=get_user_inputs)
submit_button.grid(row=36, column=0)

pilot_button = ttk.Button(root, text="Get Pilot Inputs", command=get_pilot_inputs)
pilot_button.grid(row=36, column=1)

pilot_button = ttk.Button(root, text="Submit", command=get_pilot_inputs and get_user_inputs)
pilot_button.grid(row=40, column=0)


def save_inputs_to_file():                          # Stores all the inputs in a file
    user_inputs = get_user_inputs()
    pilot_inputs = get_pilot_inputs()
    # Writes inputs to a text file
    with open("Stored_inputs.txt", "a") as file:    # 'a' mode for appending
        file.write("User Inputs:\n")
        for key, value in user_inputs.items():
            file.write(f"{key}: {value}\n")
        file.write("\nPilot Inputs:\n")
        for key, value in pilot_inputs.items():
            file.write(f"{key}: {value}\n")
        file.write("-" * 25 + "\n")                 # Run Separator 

    print("Inputs saved to file.")                  # For debugging

# Start the GUI event loop
root.mainloop()
