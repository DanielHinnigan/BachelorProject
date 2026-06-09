from femm import *
import numpy as np
import constants as constants

def coils():    
    # Allows us to change the constants using external scripts but not inside this function
    num_coils = constants.num_coils
    stator_circ = constants.stator_circ
    s = constants.l_s
    w_s1 = constants.w_s1

    # Add circuits - Currents taken from PellegrinoIPM
    mi_addcircprop("A", constants.i_A, 1)
    mi_addcircprop("C", constants.i_C, 1)
    mi_addcircprop("B", constants.i_B, 1)

    free_space = (stator_circ-num_coils*s)/num_coils

    # Draw teeth outline
    p1s = []
    p2s = []
    for c in range(num_coils):
        temp = (free_space+s)*c
        theta = temp/constants.inner_R_stator

        # Draw teeth
        x1 = constants.inner_R_stator*np.cos(theta)
        y1 = constants.inner_R_stator*np.sin(theta)

        theta += s/constants.inner_R_stator
        x2 = constants.inner_R_stator*np.cos(theta)
        y2 = constants.inner_R_stator*np.sin(theta)

        mi_drawarc(x1, y1, x2, y2, s/constants.inner_R_stator/(2*np.pi)*360, 1)
        
        p1s.append((x1,y1)) # Left point
        p2s.append((x2,y2)) # Right point
        
    # Draw coils
    for c in range(num_coils):
        # Vector going from one opening to the other opening
        p1 = np.array(p1s[c])
        p2 = np.array(p2s[c-1])

        v = p1-p2

        # Find the perpendicular vectors w.r.t the points, scale and translate to start at the point
        # This is the "indentation region"
        h = constants.h_s0
        x = h*v[1]/np.linalg.norm(v)
        y = -h*v[0]/np.linalg.norm(v)

        v1 = np.array([p1[0]+x, p1[1]+y])
        v2 = np.array([p2[0]+x, p2[1]+y])

        mi_drawline(p1[0], p1[1], v1[0], v1[1])
        mi_drawline(p2[0], p2[1], v2[0], v2[1])
        mi_drawline(v1[0], v1[1], v2[0], v2[1]) # This line is needed for separating the coil and air region

        # Create coil region
            # Extension in height
        h = constants.h_s1+constants.h_s0       
        x_h = h*v[1]/np.linalg.norm(v)
        y_h = -h*v[0]/np.linalg.norm(v)

            # Extension in width: w is for width and r is for right side of coil
        x_wr = w_s1*y/np.sqrt(y**2+x**2)
        y_wr = -w_s1*x/np.sqrt(y**2+x**2)
        v4 = np.array([x_wr, y_wr])+v2
        mi_drawline(v2[0], v2[1], v4[0], v4[1])

        x_wl = -w_s1*y/np.sqrt(y**2+x**2)
        y_wl = w_s1*x/np.sqrt(y**2+x**2)
        v3 = np.array([x_wl, y_wl])+v1
        mi_drawline(v1[0], v1[1], v3[0], v3[1])

            # Draw the rest of the coil box
        mi_drawline(v3[0], v3[1], v3[0]+x_h, v3[1]+y_h)
        mi_drawline(v4[0], v4[1], v4[0]+x_h, v4[1]+y_h)
        mi_drawline(v3[0]+x_h, v3[1]+y_h, v4[0]+x_h, v4[1]+y_h)

        # Place circuit
        circuit_x = (v3[0]+v3[0]+x_h+v4[0]+v4[0]+x_h)/4
        circuit_y = (v3[1]+v3[1]+y_h+v4[1]+v4[1]+y_h)/4
        mi_addblocklabel(circuit_x, circuit_y)
        mi_selectlabel(circuit_x, circuit_y)
        dir = 1 if constants.winding_pattern[c%len(constants.winding_pattern)][1] == "+" else -1
        mi_setblockprop("Winding", 1, 0, constants.winding_pattern[c%len(constants.winding_pattern)][0], 0, 0, constants.n_turns*dir)