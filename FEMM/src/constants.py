import numpy as np

# Motor constants [mm]
length = 60

# Coils
n_turns = 8
w_air_scaling = 0.4*0.5179

# Pattern for windings
winding_pattern = np.tile(np.flip(np.array([("A", "-"), ("A", "-"), ("C", "+"), ("C", "+"), ("B", "-"), ("B", "-"), ("A", "+"), ("A", "+"), ("C", "-"), ("C", "-"), ("B", "+"), ("B", "+")]), axis=0),(4,1))

# Rotor constants
outer_R_rotor = 140.4/2
inner_R_rotor = 47/2

# Magnets
pm_thickness = 3
pm_width = 16
theta_pm_deg = 35 # Range 10 to 35 for num_poles = 8
theta_pm = theta_pm_deg/360*2*np.pi*0.89857089

# Pole
num_poles = 8
theta_pole = 2*np.pi/num_poles

# Currents
I_s = 200 # Peak current (Toyota Prius, Fourth generation, table 1)

beta = 0.7
I_d = -I_s*np.cos(beta)
I_q =I_s*np.sin(beta)

i_A = I_d-I_q
i_B = np.cos(-2*np.pi/3)*I_d-np.sin(-2*np.pi/3)*I_q
i_C = np.cos(2*np.pi/3)*I_d-np.sin(2*np.pi/3)*I_q

# NO LOAD
"""i_A = 0
i_B = 0
i_C = 0"""

R_A = 52 # Radius of inner points of magnets
arc_length = 3 # Arc length between inner points of magnets w.r.t the centrum of the motor and the radius R_A

# Stator constants
g = 0.65 # Air gap length (mm) - Equal to the Toyata Model
inner_R_stator = outer_R_rotor+g
outer_R_stator = 215/2

# Slot constants
num_coils = len(winding_pattern)
stator_circ = 2*np.pi*inner_R_stator # Inner Circumference of stator
w_s0 = 4 # [mm]
l_s = stator_circ/num_coils-w_s0 # Length of each teeth (slot opening is the negated version)
h_s0 = 3.0 # [mm]
h_s1 = 15. # [mm]
w_s1 = 0.5 # [mm]

# Rotational angle is calculated based on the above values
theta_rot_d = -(4.5*stator_circ/num_coils/inner_R_stator+theta_pole/2*3+1/2*w_s0/inner_R_stator)*180/np.pi
theta_rot_q = theta_rot_d-2/num_poles*90
theta_rot = theta_rot_d

# --------------------------------

# Loss function constants
k_T = 1
k_f = 1

# Optimization parameters
N_theta = 1024              # Number of points to use for the angle
num_bins_turns = N_theta         # Number of bins to use for each winding
all_parameters = {
    # Initial parameters
    "time": 1, 
    "i_A": 100,
    "i_B": 100,
    "i_C": 100, 
    "magnet length": 10, # CHECK IF VALID
    "theta_r": 0,
    "magnet strength": 1, # CHECK IF VALID
    "pole pairs": num_poles,
    "arc length": arc_length,
    "theta_pm": theta_pm,

    # Each turn function has its own turn function
    # No restriction is set on the intercombination of turns, i.e. more than one phase
    # can be present in one slot
    "turns_step_A": [0]*num_bins_turns,
    "turns_step_B": [0]*num_bins_turns,
    "turns_step_C": [0]*num_bins_turns
    }
to_tune = ["i_A", "i_B", "i_C", "pole pairs", 
           "magnet length", "arc length", "theta_pm",
           "turns_step_A", "turns_step_B", "turns_step_C"]

# Static Bounds on parameters for optimization
current_bound = [0, 240]
magnet_length_bound = (0, None)
theta_pm_bound = (0, None)
pole_pair_bound = [0, 30]
arc_length_bound = [0.01, None]
theta_pm_bound = [0.01, None]

bounds = [current_bound, current_bound, current_bound, 
          pole_pair_bound, magnet_length_bound,
          arc_length_bound, theta_pm_bound, ]

# Electrical parameters
omega_e = 60
phase = np.array([-120, 0, 120])/np.pi

# ----------------------- GA OPTIMIZER -------------
# Continous parameters = i_A, i_B, i_C, magnet length, arc length, theta_pm
num_continous = 6
# Integer parameters = turns_step and pole pairs
to_tune_ga = ["i_A", "i_B", "i_C", "magnet length", "arc length", 
              "theta_pm", "pole pairs",
           "turns_step_A", "turns_step_B", "turns_step_C"]
gene_space = [
    {"low": 0, "high": 240},    # i_A bounds
    {"low": 0, "high": 240},    # i_B bounds
    {"low": 0, "high": 240},    # i_C bounds
    {"low": 0, "high": 10e3},   # Magnet Length
    {"low": 0, "high": 10e3},   # Arc length
    {"low": 0, "high": 10},     # theta_pm
] + [range(1,101)] + [[-1,0, 1]]*num_bins_turns*3
total_genes = len(gene_space)

initial_population = [[100, 100, 100, pm_width, arc_length, 
                      theta_pm, num_poles]+[0]*num_bins_turns*3]