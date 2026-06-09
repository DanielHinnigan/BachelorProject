import numpy as np

# Rotor constants
outer_R_rotor = 140.4/2*10**(-3) # Rotor outer radius [m]
inner_R_rotor = 47/2*10**(-3)

# constants
p = 4
num_poles = p*2
theta_pm = 35/180*np.pi 
theta_pole = 2*np.pi/num_poles

L = 0.06 # Axial length of motor
pm_thickness = 0.003 # length of permanent magnet
pm_width = 0.016 # Width of PM
R_A = 0.052 # Radius of inner points of magnets
arc_length = 0.003 # Arc length between inner points of magnets w.r.t the centrum of the motor and the radius R_A
l_c = 0.003 # Used for calculating R_m
g = 0.00065 # 0.65 mm
k_w = 1 # Winding factor
N_c=8.0*2 # Number of windings per coil
coils_per_phase = 4
N_ph = N_c*coils_per_phase # Number of windings per phase
I_s = 200 # Used in octave file

A_c = 0.02324*L # Area of one magnet plus airgap
A_m = pm_width*L # Cross-sectional area of magnet

B_r = 1.0181  # Remanent B-field of PMs
B_sat = 2.6 # Saturation B-field of steel

mu0 = 4*np.pi*10**(-7)

magnet_cords = None # To be updated in Jeihoon

#0.51793674*0.4
w_air_scaling = 0.51793674*0.4
"""
A_g1 = r_R*theta_pm*L
A_g2 = r_R*(theta_p-theta_pm)/2*L
A_b = 53.85*10**(-6)*L

phi_b = A_b*B_sat
phi_m = A_m*B_r"""