import numpy as np

# Motor constants [mm]
length = 60

# Coils
n_turns = 8
w_air_scaling = 0.51793674*0.4

# Pattern for windings
winding_pattern = np.tile(np.flip(np.array([("A", "-"), ("A", "-"), ("C", "+"), ("C", "+"), ("B", "-"), ("B", "-"), ("A", "+"), ("A", "+"), ("C", "-"), ("C", "-"), ("B", "+"), ("B", "+")]), axis=0),(4,1))

# Rotor constants
outer_R_rotor = 140.4/2
inner_R_rotor = 47/2

# Magnets
pm_thickness = 3
pm_width = 16
theta_pm_deg = 35 # Range 10qqqqq to 75 for num_poles = 4
theta_pm = theta_pm_deg/360*2*np.pi*0.99309457

# Pole
num_poles = 8
theta_pole = 2*np.pi/num_poles
theta_rot_d = theta_pole/2-0.8*2*np.pi/len(winding_pattern)/2 # d-axis configuration and zero electrical degrees
theta_rot_q = theta_rot_d+90/180*np.pi/num_poles/2 # q-axis configuration

# Currents
I_s = 200 # Peak current (Toyota Prius, Fourth generation, table 1)

beta = 0.6454
I_d = -I_s*np.cos(beta)
I_q =I_s*np.sin(beta)

i_A = I_d
i_B = I_d*np.cos(120/180*np.pi)+I_q*np.sin(120/180*np.pi)
i_C = I_d*np.cos(120/180*np.pi)-I_q*np.sin(120/180*np.pi)

R_A = 52 # Radius of inner points of magnets
arc_length = 3 # Arc length between inner points of magnets w.r.t the centrum of the motor and the radius R_A

# Stator constants
g = 0.65 # Air gap length (mm) - Equal to the Toyata Model
inner_R_stator = outer_R_rotor+g
outer_R_stator = 215/2

# Slot constants
num_coils = len(winding_pattern)
stator_circ = 2*np.pi*inner_R_stator # Inner Circumference of stator
w_s0 = 1 # [mm]
l_s = stator_circ/num_coils-w_s0 # Length of each teeth (slot opening is the negated version)
h_s0 = 3 # [mm]
h_s1 = 10 # [mm]
w_s1 = 0.1 # [mm]

# Ranges [mm]
h_s0_range = (0.5, 3.0)
h_s1_range = (1.0, 30.0)
w_s0_range = (0.5, 5.0)
w_s1_range = (0.1, 2.0)

# Rotational angle is calculated based on the above values
theta_rot = -(4.5*stator_circ/num_coils/inner_R_stator+theta_pole/2*3+1/2*w_s0/inner_R_stator)*180/np.pi

# Surrogate model
err_threshold = 0.01