import sympy as sp
import constants
import numpy as np
# --------------------------------- V2 ------------------------

def calculate_magnet_points():
    # Constants
    theta_p = 2*np.pi/constants.num_poles

    # Radius of upper inner corner of magnets
    theta_temp = constants.arc_length/constants.R_A
    theta_A = (theta_p-theta_temp)/2
    
    alpha, beta, R_B = sp.symbols("alpha, beta, R_B")
    l_eq = constants.pm_width*(1+constants.w_air_scaling)
    temp = sp.solve([
        np.pi-alpha-beta-(constants.theta_pm-theta_A)/2, 
        constants.R_A/sp.sin(beta)-l_eq/sp.sin((constants.theta_pm-theta_A)/2), 
        R_B/sp.sin(alpha)-l_eq/sp.sin((constants.theta_pm-theta_A)/2)
        ], rational=False)

    temp = temp[0] if temp[0][R_B]>temp[1][R_B] else temp[1]
    R_B = float(temp[R_B])

    coordinates = []
    for p in range(constants.num_poles):
        coordinates.append({"Right": None, "Left": None})

        # Lower left corner of first magnet
        theta_temp = constants.arc_length/constants.R_A
        theta_A = (theta_p-theta_temp)/2+theta_p*p
        x_A = constants.R_A*np.cos(theta_A)
        y_A = constants.R_A*np.sin(theta_A)

        # Upper left corner of first magnet
        x_ul = R_B*np.cos((theta_p-constants.theta_pm)/2+theta_p*p)
        y_ul = R_B*np.sin((theta_p-constants.theta_pm)/2+theta_p*p)
        
        # Upper right corner of first magnet: Calculated by using the two conditions
        # (1): Vector from UL to UR must be orthogonal to UL
        # (2): Vector between UL and UR must have length equal to a predetermined value
        # (3): Vector must point the correct way
        wx, wy = sp.symbols("wx, wy")
        sol1 = np.array(sp.nsolve([(x_ul-x_A)*wx+(y_ul-y_A)*wy, constants.pm_thickness**2-wx**2-wy**2], [wx, wy], [0.001,0.001])).astype(np.float64).flatten()
        sol2 = np.array(sp.nsolve([(x_ul-x_A)*wx+(y_ul-y_A)*wy, constants.pm_thickness**2-wx**2-wy**2], [wx, wy], [-0.001,-0.001])).astype(np.float64).flatten()
        sol = sol1 if (np.arctan2(sol1[1]+y_ul,sol1[0]+x_ul)%(2*np.pi)) < (np.arctan2(y_ul,x_ul)%(2*np.pi)) else sol2

        x_ur = x_ul + sol[0]
        y_ur = y_ul + sol[1]

        # Lower right corner
        x_lr = x_A + sol[0]
        y_lr = y_A + sol[1]

        coordinates[p]["Right"] = {
            "UL": [x_ul, y_ul],
            "UR": [x_ur, y_ur],
            "LR": [x_lr, y_lr],
            "LL": [x_A, y_A]
        }

        # --------------------------------- Second Magnet ---------------------------------------
        
        # Lower right corner of second magnet
        theta_temp = constants.arc_length/constants.R_A
        theta_A = (theta_p+theta_temp)/2+theta_p*p
        x_A = constants.R_A*np.cos(theta_A)
        y_A = constants.R_A*np.sin(theta_A)

        # Upper right corner of second magnet
        x_ur = R_B*np.cos((theta_p+constants.theta_pm)/2+theta_p*p)
        y_ur = R_B*np.sin((theta_p+constants.theta_pm)/2+theta_p*p)
        
        # Upper left corner of second magnet
        wx, wy = sp.symbols("wx, wy")
        sol1 = np.array(sp.nsolve([(x_ur-x_A)*wx+(y_ur-y_A)*wy, constants.pm_thickness**2-wx**2-wy**2], [wx, wy], [0.001,0.001])).astype(np.float64).flatten()
        sol2 = np.array(sp.nsolve([(x_ur-x_A)*wx+(y_ur-y_A)*wy, constants.pm_thickness**2-wx**2-wy**2], [wx, wy], [-0.001,-0.001])).astype(np.float64).flatten()
        sol = sol1 if (np.arctan2(sol1[1]+y_ur,sol1[0]+x_ur)%(2*np.pi)) > (np.arctan2(y_ur,x_ur)%(2*np.pi)) else sol2

        x_ul = x_ur + sol[0]
        y_ul = y_ur + sol[1]

        # Lower left corner
        x_ll = x_A + sol[0]
        y_ll = y_A + sol[1]

        # Coordinates in [m]
        coordinates[p]["Left"] = {
            "UL": [x_ul, y_ul],
            "UR": [x_ur, y_ur],
            "LR": [x_A, y_A],
            "LL": [x_ll, y_ll]
        }
    return coordinates