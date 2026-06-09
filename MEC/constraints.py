import numpy as np
import sympy as sp
import constants

from permanent_magnet import calculate_magnet_points

# -------------------- V8 -------------------------------------------------
def magnet_length_constraint_func(x):
    # x is a irrelevant parameter but needed by scipy
    constants.magnet_cords = calculate_magnet_points()

    # Coordinates of first pole right magnet
    coordinates = constants.magnet_cords[0]["Right"]
    x_A, y_A = coordinates["LL"]
    x_B, y_B = coordinates["UL"]

    # The length of the magnet must not increase beyond delta.
    # This is equivalent to having a positive delta
    a, b, x, y = sp.symbols("a, b, x, y")

    # Solve equations to find maximum allowable increase in the length of the magnets
    temp = sp.nsolve(
        [x_A*a+b-y_A, x_B*a+b-y_B, x*a+b-y, constants.outer_R_rotor**2-x**2-y**2], 
        [a, b, x, y],
        [1.0, -0.1, constants.outer_R_rotor, 0.0])
        
    # Vector from UL to intersection
    x = float(temp[2])
    y = float(temp[3])

    inter = np.array([x, y])
    ul = np.array(coordinates["UL"])
    vec = inter-ul


    # If vec is parallel to the vector from LL to UL, then delta > 0. If vec is antiparallel, then delta < 0
    delta = 0
    ll = np.array(coordinates["LL"])
    if np.round(np.vdot(vec, ul-ll)/np.linalg.norm(vec)/np.linalg.norm(ul-ll),1) == 1.0:
        # Parallel
        delta = np.linalg.norm(vec)
    else:
        # Antiparallel
        delta = -np.linalg.norm(vec)

    # Minimum distance from outer rotor: 0.2 mm
    a = 0.0002

    # Return
    return (delta-a)*5000

def theta_pm_constraint_func(x):
    # Pole angle
    theta_p = 2*np.pi/constants.num_poles

    # Coordinates of magnet
    coordinates = constants.magnet_cords[0]["Right"]
    ur = np.array(coordinates["UR"])
    ul = np.array(coordinates["UL"])

    """# Calculate terms
    term1 = (theta_p-constants.theta_pm)/2
    term2 = np.vdot(ur,ul)/np.linalg.norm(ur)/np.linalg.norm(ul)"""

    R_vec = np.array((constants.outer_R_rotor, 0))
    proj = np.vdot(R_vec, ur)/constants.outer_R_rotor**2*R_vec
    delta = np.linalg.norm(ur-proj)

    # Distance from pole function is 2.5 mm
    a = 0.0025

    # Return (needs to be greater than 0)
    return (delta-a)*1000

# Constraint
constraints = (
    {
        "type": "ineq",
        "fun": magnet_length_constraint_func
    },
    {
        "type": "ineq",
        "fun": theta_pm_constraint_func
    }
)