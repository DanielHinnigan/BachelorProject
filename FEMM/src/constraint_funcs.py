from src.constants import *
from src.permanent_magnet import calculate_magnet_points

import numpy as np
import sympy as sp

# WINDING FUNCTION CONSTRAINT OR AT LEAST IMPLEMENTATION!!!!!

def magnet_length_constraint_func(x):
    # Coordinates of first pole right magnet
    coordinates = calculate_magnet_points()[0]["Right"]
    x_A, y_A = coordinates["LL"]
    x_B, y_B = coordinates["UL"]

    # The length of the magnet must not increase beyond delta.
    # This is equivalent to having a positive delta
    delta, a, b, x, y = sp.symbols("delta, a, b, x, y")

    # Solve equations to find maximum allowable increase in the length of the magnets
    temp = sp.solve([x_A*a+b-y_A, x_B*a+b-y_B, x*a+b-y, outer_R_rotor**2-x**2-y**2, delta-y+y_B], rational=False)
    delta = float(temp[0][delta])

    return delta

def theta_pm_constraint_func(x):
    # Pole angle
    theta_p = 2*np.pi/num_poles

    # Coordinates of magnet
    coordinates = calculate_magnet_points()[0]["Right"]
    ur = np.array(coordinates["UR"])
    ul = np.array(coordinates["UL"])

    # Calculate terms
    term1 = (theta_p-theta_pm)/2
    term2 = np.vdot(ur,ul)/np.linalg.norm(ur)/np.linalg.norm(ul)

    return term1-term2

# Dynamic bounds (contraints)
# Maybe also include constraint on shape of torque curve?
magnet_length_constraint = {"type": "ineq", "fun": magnet_length_constraint_func}
theta_pm_constraint = {"type": "ineq", "fun": theta_pm_constraint_func}

constraints = (magnet_length_constraint, theta_pm_constraint)