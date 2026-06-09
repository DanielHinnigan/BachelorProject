from femm import *
import sympy as sp
from constants import *
import constants as constants
import numpy as np

def calculate_magnet_points():
    # Constants
    theta_p = 2*np.pi/num_poles

    # Radius of upper inner corner of magnets
    theta_temp = arc_length/R_A
    theta_A = (theta_p-theta_temp)/2
    
    alpha, beta, R_B = sp.symbols("alpha, beta, R_B")
    temp = sp.solve([
        np.pi-alpha-beta-(constants.theta_pm-theta_A)/2, 
        R_A/sp.sin(beta)-pm_width/sp.sin((constants.theta_pm-theta_A)/2), 
        R_B/sp.sin(alpha)-pm_width/sp.sin((constants.theta_pm-theta_A)/2)], rational=False)

    temp = temp[0] if temp[0][R_B]>temp[1][R_B] else temp[1]
    R_B = float(temp[R_B])

    coordinates = []
    for p in range(num_poles):
        coordinates.append({"Right": None, "Left": None})

        # Lower left corner of first magnet
        theta_temp = arc_length/R_A
        theta_A = (theta_p-theta_temp)/2+theta_p*p
        x_A = R_A*np.cos(theta_A)
        y_A = R_A*np.sin(theta_A)

        # Upper left corner of first magnet
        x_ul = R_B*np.cos((theta_p-constants.theta_pm)/2+theta_p*p)
        y_ul = R_B*np.sin((theta_p-constants.theta_pm)/2+theta_p*p)
        
        # Upper right corner of first magnet
        wx, wy = sp.symbols("wx, wy")
        sol1 = np.array(sp.nsolve([(x_ul-x_A)*wx+(y_ul-y_A)*wy, pm_thickness**2-wx**2-wy**2], [wx, wy], [1,1])).astype(np.float64).flatten()
        sol2 = np.array(sp.nsolve([(x_ul-x_A)*wx+(y_ul-y_A)*wy, pm_thickness**2-wx**2-wy**2], [wx, wy], [-1,-1])).astype(np.float64).flatten()
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
        theta_temp = arc_length/R_A
        theta_A = (theta_p+theta_temp)/2+theta_p*p
        x_A = R_A*np.cos(theta_A)
        y_A = R_A*np.sin(theta_A)

        # Upper right corner of second magnet
        x_ur = R_B*np.cos((theta_p+constants.theta_pm)/2+theta_p*p)
        y_ur = R_B*np.sin((theta_p+constants.theta_pm)/2+theta_p*p)
        
        # Upper left corner of second magnet
        wx, wy = sp.symbols("wx, wy")
        sol1 = np.array(sp.nsolve([(x_ur-x_A)*wx+(y_ur-y_A)*wy, pm_thickness**2-wx**2-wy**2], [wx, wy], [1,1])).astype(np.float64).flatten()
        sol2 = np.array(sp.nsolve([(x_ur-x_A)*wx+(y_ur-y_A)*wy, pm_thickness**2-wx**2-wy**2], [wx, wy], [-1,-1])).astype(np.float64).flatten()
        sol = sol1 if (np.arctan2(sol1[1]+y_ur,sol1[0]+x_ur)%(2*np.pi)) > (np.arctan2(y_ur,x_ur)%(2*np.pi)) else sol2

        x_ul = x_ur + sol[0]
        y_ul = y_ur + sol[1]

        # Lower left corner
        x_ll = x_A + sol[0]
        y_ll = y_A + sol[1]

        coordinates[p]["Left"] = {
            "UL": [x_ul, y_ul],
            "UR": [x_ur, y_ur],
            "LR": [x_A, y_A],
            "LL": [x_ll, y_ll]
        }

    return coordinates


# Uses angle between upper inner points and the arc length defined by the inner lower points
def permanent_magnets_v2():
    coordinates = calculate_magnet_points()

    for p in range(num_poles):
        c = coordinates[p]["Right"]
        x_ur, y_ur = c["UR"]
        x_ul, y_ul = c["UL"]
        x_lr, y_lr = c["LR"]
        x_ll, y_ll = c["LL"]

        # Draw First magnet
        mi_drawline(x_ur, y_ur, x_ul, y_ul)
        mi_drawline(x_ur, y_ur, x_lr, y_lr)
        mi_drawline(x_lr,y_lr,x_ll,y_ll)
        mi_drawline(x_ll,y_ll,x_ul,y_ul)

        # Add material to magnet
        midpoint_x = (x_ur+x_ul+x_lr+x_ll)/4
        midpoint_y = (y_ur+y_ul+y_lr+y_ll)/4

        mi_addblocklabel(midpoint_x, midpoint_y)
        mi_selectlabel(midpoint_x, midpoint_y)

        # Check if north or south pole and choose direction thereon
        if (p%2) == 0:
            # North pole
            mag_dir = np.arctan2(y_ul-y_ur, x_ul-x_ur)
            mi_setblockprop("42SH", 0, 0, 0, mag_dir/(2*np.pi)*360, 1)
        else:
            mag_dir = np.arctan2(-y_ul+y_ur, -x_ul+x_ur)
            mi_setblockprop("42SH", 0, 0, 0, mag_dir/(2*np.pi)*360, 1)

        # Add outer flux-barrier
        mi_drawline(x_ul, y_ul, x_ul+constants.w_air_scaling*(x_ul-x_ll), y_ul+constants.w_air_scaling*(y_ul-y_ll))
        mi_drawline(x_ur, y_ur, x_ur+constants.w_air_scaling*(x_ur-x_lr), y_ur+constants.w_air_scaling*(y_ur-y_lr))
        mi_drawline(x_ul+constants.w_air_scaling*(x_ul-x_ll), y_ul+constants.w_air_scaling*(y_ul-y_ll), x_ur+constants.w_air_scaling*(x_ur-x_lr), y_ur+constants.w_air_scaling*(y_ur-y_lr))
        
        midpoint_x = (x_ul+x_ur+x_ul+constants.w_air_scaling*(x_ul-x_ll)+x_ur+constants.w_air_scaling*(x_ur-x_lr))/4
        midpoint_y = (y_ul+y_ur+y_ul+constants.w_air_scaling*(y_ul-y_ll)+y_ur+constants.w_air_scaling*(y_ur-y_lr))/4
        mi_addblocklabel(midpoint_x, midpoint_y)
        mi_selectlabel(midpoint_x, midpoint_y)
        mi_setblockprop("Air")
        mi_setgroup(2)

        # ------------------------------------ Second Magnet ------------------------------------
        c = coordinates[p]["Left"]
        x_ur, y_ur = c["UR"]
        x_ul, y_ul = c["UL"]
        x_lr, y_lr = c["LR"]
        x_ll, y_ll = c["LL"]

        # Draw second magnet
        mi_drawline(x_ur, y_ur, x_ul, y_ul)
        mi_drawline(x_ur, y_ur, x_lr, y_lr)
        mi_drawline(x_ll,y_ll,x_lr,y_lr)
        mi_drawline(x_ll,y_ll,x_ul,y_ul)

        # Add material to magnet
        midpoint_x = (x_ur+x_ul+x_ll+x_lr)/4
        midpoint_y = (y_ur+y_ul+y_ll+y_lr)/4

        mi_addblocklabel(midpoint_x, midpoint_y)
        mi_selectlabel(midpoint_x, midpoint_y)

        # Check if north or south pole and choose direction thereon
        if (p%2) == 0:
            # North pole
            mag_dir = np.arctan2(y_ur-y_ul, x_ur-x_ul)
            mi_setblockprop("42SH", 0, 0, 0, mag_dir/(2*np.pi)*360, 1)
        else:
            mag_dir = np.arctan2(y_ul-y_ur, x_ul-x_ur)
            mi_setblockprop("42SH", 0, 0, 0, mag_dir/(2*np.pi)*360, 1)


        # Add outer flux-barrier
        mi_drawline(x_ul, y_ul, x_ul+constants.w_air_scaling*(x_ul-x_ll), y_ul+constants.w_air_scaling*(y_ul-y_ll))
        mi_drawline(x_ur, y_ur, x_ur+constants.w_air_scaling*(x_ur-x_lr), y_ur+constants.w_air_scaling*(y_ur-y_lr))
        mi_drawline(x_ul+constants.w_air_scaling*(x_ul-x_ll), y_ul+constants.w_air_scaling*(y_ul-y_ll), x_ur+constants.w_air_scaling*(x_ur-x_lr), y_ur+constants.w_air_scaling*(y_ur-y_lr))
        
        midpoint_x = (x_ul+x_ur+x_ul+constants.w_air_scaling*(x_ul-x_ll)+x_ur+constants.w_air_scaling*(x_ur-x_lr))/4
        midpoint_y = (y_ul+y_ur+y_ul+constants.w_air_scaling*(y_ul-y_ll)+y_ur+constants.w_air_scaling*(y_ur-y_lr))/4
        mi_addblocklabel(midpoint_x, midpoint_y)
        mi_selectlabel(midpoint_x, midpoint_y)
        mi_setblockprop("Air")
        mi_setgroup(2)

        # -------------------------- FLUX BARRIER BETWEEN MAGNETS -------------------------
        c_left = coordinates[p]["Left"]
        c_right = coordinates[p]["Right"]

        mi_drawline(c_left["LR"][0], c_left["LR"][1], c_right["LL"][0], c_right["LL"][1])
        mi_drawline(c_left["LL"][0], c_left["LL"][1], c_right["LR"][0], c_right["LR"][1])

        midpoint_x = (c_left["LR"][0]+c_left["LL"][0]+c_right["LL"][0]+c_right["LR"][0])/4
        midpoint_y = (c_left["LR"][1]+c_left["LL"][1]+c_right["LL"][1]+c_right["LR"][1])/4
        mi_addblocklabel(midpoint_x, midpoint_y)
        mi_selectlabel(midpoint_x, midpoint_y)
        mi_setblockprop("Air")
        mi_setgroup(2)

        mi_selectcircle(0,0,outer_R_rotor, 0)
        mi_setgroup(2)