import numpy as np
import sympy as sp

import constants
from permanent_magnet import *

# --------------------- V1 ---------------------------------
def bridge_current():
    # Left magnet
    left = constants.magnet_cords[0]["Left"]

    # Find linear function
    a = (left["UL"][1]-left["LL"][1])/(left["UL"][0]-left["LL"][0])
    b = (left["UL"][0]*left["LL"][1]-left["UL"][1]*left["LL"][0])/(left["UL"][0]-left["LL"][0])
 
    # Solve for delta
    delta, x, y = sp.symbols("delta, x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LL"][0] > 0 and temp[0][y]*left["LL"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta = np.sqrt(float((temp[x]-left["UL"][0])**2+(temp[y]-left["UL"][1])**2))

    # Calculate bridge cross sectional area
    A_b = delta*constants.L

    return A_b*constants.B_sat

# Reluctance behind magnet (from airgap to magnet)
def r_y1(mu_y1):
    UR_y = constants.magnet_cords[0]["Right"]["UR"][1]
    A_y1 = UR_y*constants.L

    return constants.pm_width/mu_y1/A_y1, A_y1

# Reluctance between magnets in pole
def r_y2(mu_y2):
    # Get magnet coordinates
    cords = constants.magnet_cords[0]
    right = cords["Right"]
    left = cords["Left"]

    # Length (look in document)
    UL_right = right["UL"]
    UR_left = left["UR"]

    L_y2 = np.sqrt((UL_right[0]-UR_left[0])**2+(UL_right[1]-UR_left[1])**2)

    # Cross sectional area
    A_y2 = L_y2*constants.L

    # Width: Slight cheat: We'll say that the width is inverse to the pm angle by some starting magnitude equal to the length of the magnet
    # NOTE: 10 is the minimum PM angle used
    theta_pm = constants.theta_pm/np.pi*180 # Convert to angles
    w_y2 = 10/theta_pm*constants.pm_width

    return w_y2/mu_y2/A_y2, A_y2

def r_g2():
    # Angle covering the second airgap
    alpha_0 = (constants.theta_pole-constants.theta_pm)/2

    # Cross-sectional area of second airgap
    A_g2 = constants.outer_R_rotor*alpha_0*constants.L

    return constants.g/constants.mu0/A_g2, A_g2

def r_g1_d_axis():
    # Cross-sectional area of second airgap
    A_g1 = constants.outer_R_rotor*constants.theta_pm*constants.L

    return constants.g/constants.mu0/A_g1

def r_r(mu_r):
    # Left magnet
    left = constants.magnet_cords[0]["Left"]

    # Find linear function
    a = (left["UL"][1]-left["LL"][1])/(left["UL"][0]-left["LL"][0])
    b = (left["UL"][0]*left["LL"][1]-left["UL"][1]*left["LL"][0])/(left["UL"][0]-left["LL"][0])

    # Solve for delta
    delta, x, y = sp.symbols("delta, x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LL"][0] > 0 and temp[0][y]*left["LL"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta = np.sqrt(float((temp[x]-left["UL"][0])**2+(temp[y]-left["UL"][1])**2))

    # Calculate bridge cross sectional area
    A_r = delta*constants.L

    return constants.pm_thickness/mu_r/A_r, A_r

def r_g1_y_axis():
    # Cross-sectional area of second airgap
    A_g1 = constants.outer_R_rotor*constants.theta_pm/2*constants.L

    return 2*np.log((constants.outer_R_rotor+constants.g)/constants.outer_R_rotor)/(constants.theta_pm*constants.L*constants.mu0), A_g1

    return constants.g/constants.mu0/A_g1

# --------------------- V3 ------------------------------------
def r_r_v3(mu_r):
    # Left magnet
    left = constants.magnet_cords[0]["Left"]

    # ---------------- delta1 --------------
    # Find linear function for delta1
    a = (left["UL"][1]-left["LL"][1])/(left["UL"][0]-left["LL"][0])
    b = (left["UL"][0]*left["LL"][1]-left["UL"][1]*left["LL"][0])/(left["UL"][0]-left["LL"][0])

    # Solve for delta
    x, y = sp.symbols("x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LL"][0] > 0 and temp[0][y]*left["LL"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta1 = np.sqrt(float((temp[x]-left["UL"][0])**2+(temp[y]-left["UL"][1])**2))

    # ---------------- delta2 --------------
    # Find linear function for delta2
    a = (left["UR"][1]-left["LR"][1])/(left["UR"][0]-left["LR"][0])
    b = (left["UR"][0]*left["LR"][1]-left["UR"][1]*left["LR"][0])/(left["UR"][0]-left["LR"][0])

    # Solve for delta
    x, y = sp.symbols("x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LR"][0] > 0 and temp[0][y]*left["LR"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta2 = np.sqrt(float((temp[x]-left["UR"][0])**2+(temp[y]-left["UR"][1])**2))
    # --------------- Cross-sectional area -----------------

    # Calculate bridge cross sectional area
    A_r = (delta1+delta2)/2*constants.L
    A_r = delta2*constants.L

    return constants.pm_thickness/mu_r/A_r, A_r

# --------------------- V5 ---------------------------------
# Updated version does not calculate the bridge flux using B_sat
# Instead the estimated B field is used. The B-field is 
def bridge_current_v5(B_b):
    # Left magnet
    left = constants.magnet_cords[0]["Left"]

    # ---------------- delta1 --------------
    # Find linear function for delta1
    a = (left["UL"][1]-left["LL"][1])/(left["UL"][0]-left["LL"][0])
    b = (left["UL"][0]*left["LL"][1]-left["UL"][1]*left["LL"][0])/(left["UL"][0]-left["LL"][0])

    # Solve for delta
    x, y = sp.symbols("x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LL"][0] > 0 and temp[0][y]*left["LL"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta1 = np.sqrt(float((temp[x]-left["UL"][0])**2+(temp[y]-left["UL"][1])**2))

    # ---------------- delta2 --------------
    # Find linear function for delta2
    a = (left["UR"][1]-left["LR"][1])/(left["UR"][0]-left["LR"][0])
    b = (left["UR"][0]*left["LR"][1]-left["UR"][1]*left["LR"][0])/(left["UR"][0]-left["LR"][0])

    # Solve for delta
    x, y = sp.symbols("x, y")
    temp = sp.solve([x**2+y**2-constants.outer_R_rotor**2, a*x+b-y], rational=False)

    # Two solutions to temp: One is parallel to UL and LL ; the other is anti parallel (shifted by 180 degrees)
    if temp[0][x]*left["LR"][0] > 0 and temp[0][y]*left["LR"][1] > 0:
        temp = temp[0]
    else:
        temp = temp[1]
    delta2 = np.sqrt(float((temp[x]-left["UR"][0])**2+(temp[y]-left["UR"][1])**2))

    # --------------- Cross-sectional area -----------------

    # Calculate bridge cross sectional area
    A_b = (delta1+delta2)/2*constants.L
    A_b = delta2*constants.L

    return A_b*B_b
# ----------------------- V6 D-axis -----------------------
def r_a_d_v6():
    # Get usefull coordinates
    cords = constants.magnet_cords[0]
    right_LL = np.array(cords["Right"]["LL"])
    left_LR = np.array(cords["Left"]["LR"])
    right_LR = np.array(cords["Right"]["LR"])

    # Calculate
    w_a = np.linalg.norm(right_LL-left_LR)
    l_a = np.linalg.norm(right_LL-right_LR)

    # Cross-sectional area
    A_a = w_a*constants.L

    return l_a/constants.mu0/A_a, A_a

def r_y3_d_v6(mu_y3):
    left = constants.magnet_cords[0]["Left"]
    right = constants.magnet_cords[0]["Right"]

    left_LR = np.array(left["LR"])
    right_LL = np.array(right["LL"])

    # Center of lower part of airgap connecting the two magnets
    center = (left_LR+right_LL)/2

    # Width of cross section
    w_y3 = np.linalg.norm(center)-constants.inner_R_rotor

    # Length of reluctance element is equal to the length of air element
    l_y3 = np.linalg.norm(left_LR-right_LL)

    # Area
    A_y3 = w_y3*constants.L

    return l_y3/A_y3/mu_y3, A_y3

def r_g_d_v6():
    # Cross-sectional area of second airgap
    A_g1 = constants.outer_R_rotor*constants.theta_pm/2*constants.L

    return np.log((constants.outer_R_rotor+constants.g)/constants.outer_R_rotor)/(constants.theta_pm/2*constants.L*constants.mu0), A_g1

# Reluctance between magnets inbetween two poles
def r_y2_d_v6(mu_y2):
    UR_y = np.abs(constants.magnet_cords[0]["Right"]["UR"][1])
    LR_y = np.abs(constants.magnet_cords[0]["Right"]["LR"][1])
    A_y2 = (UR_y+LR_y)/2*constants.L*2

    UR = np.array(constants.magnet_cords[0]["Right"]["UR"])
    UL = np.array(constants.magnet_cords[-1]["Left"]["UL"])

    w = np.linalg.norm(UR-UL)

    A_y2 = constants.pm_width*constants.L

    return w/A_y2/mu_y2, A_y2

    return constants.pm_thickness/mu_y2/A_y2, A_y2

# Reluctance between magnets in pole
def r_y1_d_v6(mu_y1):
    # Get magnet coordinates
    cords = constants.magnet_cords[0]
    right = cords["Right"]
    left = cords["Left"]

    # Length (look in document)
    UL_right = right["UL"]
    UR_left = left["UR"]

    L_y2 = np.sqrt((UL_right[0]-UR_left[0])**2+(UL_right[1]-UR_left[1])**2)/2

    # Cross sectional area
    A_y2 = L_y2*constants.L

    # Width: Slight cheat: We'll say that the width is inverse to the pm angle by some starting magnitude equal to the length of the magnet
    # NOTE: 10 is the minimum PM angle used
    theta_pm = constants.theta_pm/np.pi*180 # Convert to angles
    w_y2 = 10/theta_pm*constants.pm_width

    return w_y2/mu_y1/A_y2, A_y2