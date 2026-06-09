from femm import *
import os
import json 

from permanent_magnet import *
import constants
from coils import *
from materials import add_materials

from concurrent.futures import ProcessPoolExecutor
from smt.sampling_methods import LHS

rot_angle = 10 # [degree]
# Open FEMM
# Iterate over theta pms
poles_pairs = num_poles/2

def construct_stator():
    outer_R_stator = constants.outer_R_stator

    # Outer stator
    mi_addnode(outer_R_stator,0)
    mi_addnode(-outer_R_stator,0)

    mi_addarc(outer_R_stator,0,-outer_R_stator,0,180,1)
    mi_addarc(-outer_R_stator,0,outer_R_stator,0,180,1)

    # Material - Stator back iron
    mi_addblocklabel(outer_R_stator-0.01, 0)
    mi_selectlabel(outer_R_stator-0.01, 0)
    mi_setblockprop("M-19 Steel")
    mi_clearselected()
    
    # Place coils
    coils()

# Build rotor
def rotor():
    outer_R_rotor = constants.outer_R_rotor
    inner_R_rotor = constants.inner_R_rotor

    # Outer rotor
    mi_addnode(outer_R_rotor,0)
    mi_addnode(-outer_R_rotor,0)

    mi_addarc(outer_R_rotor,0,-outer_R_rotor,0,180,1)
    mi_addarc(-outer_R_rotor,0,outer_R_rotor,0,180,1)

    # Inner rotor
    mi_addnode(inner_R_rotor,0)
    mi_addnode(-inner_R_rotor,0)

    mi_addarc(inner_R_rotor,0,-inner_R_rotor,0,180,1)
    mi_addarc(-inner_R_rotor,0,inner_R_rotor,0,180,1)

    # Material - Inner rotor
    mi_addblocklabel(0, 0)
    mi_selectlabel(0, 0)
    mi_setblockprop("Air")
    mi_clearselected()

    # Material - Outer rotor
    mi_addblocklabel(inner_R_rotor+0.01, 0)
    mi_selectlabel(inner_R_rotor+0.01, 0)
    mi_setblockprop("M-19 Steel")
    mi_setgroup(1)
    mi_clearselected()

    # Material - Air gap
    mi_addblocklabel(outer_R_rotor+0.01, 0)
    mi_selectlabel(outer_R_rotor+0.01, 0)
    mi_setblockprop("Air")
    mi_clearselected()

def analyze_model():
    # Analyse
    mi_analyze(1)
    mi_loadsolution()

    # Calculate Torque
    mo_groupselectblock(1)
    torque = mo_blockintegral(22)
    mo_clearblock()

    # Close window
    mo_close()

    return torque

# Worker function
def analyze_rot(rot, h_s0, h_s1, w_s0, w_s1):
    # Update the constants in the global constants file
    constants.h_s0 = h_s0
    constants.h_s1 = h_s1
    constants.w_s0 = w_s0
    constants.w_s1 = w_s1

    # Open FEMM
    openfemm(1)

    # Create a new document
    newdocument(0)
    main_maximize()
    mi_probdef(0, 'millimeters', 'planar', 1e-8, length, 30)

    # Add materials
    add_materials()

    # Material outside motor
    mi_addblocklabel(outer_R_stator+1, 0)
    mi_selectlabel(outer_R_stator+1, 0)
    mi_setblockprop("Air")
    mi_clearselected()

    # Build rotor
    rotor()

    # Build stator
    construct_stator()

    # Insert magnets
    permanent_magnets_v2()
    
    # Zoom to fit
    mi_zoomnatural()

    # Boundaries
    mi_makeABC(7, constants.outer_R_stator+10, 0, 0, 0)

    # Rotate to align with the d-axis
    mi_selectgroup(1)
    mi_selectgroup(2)
    mi_moverotate(0,0,constants.theta_rot)
    
    # Rotate rotor w.r.t the zero-degree electrical degrees
    if rot > 0:
        mi_selectgroup(1)
        mi_selectgroup(2)
        mi_moverotate(0, 0, -rot)

    # Change currents
    I_d = constants.I_d
    I_q = constants.I_q

    I_a = I_d*np.cos(np.radians(poles_pairs*rot))-I_q*np.sin(np.radians(poles_pairs*rot))
    I_b = I_d*np.cos(np.radians(poles_pairs*rot-120))-I_q*np.sin(np.radians(poles_pairs*rot-120))
    I_c = I_d*np.cos(np.radians(poles_pairs*rot+120))-I_q*np.sin(np.radians(poles_pairs*rot+120))

    mi_modifycircprop('A', 1, I_a)
    mi_modifycircprop('B', 1, I_b)
    mi_modifycircprop('C', 1, I_c)

    # Save file
    file = f"file{rot}.FEM"

    if os.path.exists(file):
            os.remove(file)
    mi_saveas(file)

    # Analyze
    T = analyze_model()

    # Close femm
    closefemm()

    # Logging
    print(f"Rotational angle: {rot:.2f} ; Torque: {T:.2f}")

    return rot, T

# Create a motor model using h_s0, h_s1, w_s0, w_s1
# Look in "Motor Design.docx" for explanation of quantities
def create_model(h_s0, h_s1, w_s0, w_s1):    
    # Iterate over rotational angles
    torques = []
    rot_angles = []

    # Run workers
    max_workers = 1
    rot_angles = np.arange(0.0, 91.0, rot_angle)

    # Add randomness: Each angle but the last (90 degrees) is added a random digit between 0 and 4.
    # This is done, since unusually ordered pattern is observed when only using a 10 degree rot_angle
    for i in range(len(rot_angles)-1):
        rot_angles[i] += np.random.uniform(0,9)

    with ProcessPoolExecutor(max_workers=max_workers) as e:
        results = e.map(analyze_rot, rot_angles,
                        [h_s0]*len(rot_angles),
                        [h_s1]*len(rot_angles),
                        [w_s0]*len(rot_angles),
                        [w_s1]*len(rot_angles))
    print(results)
    results = np.array(list(results))
    print(results)

    rot_angles = results[:,0]
    torques = results[:,1]

    return rot_angles, torques

if __name__ == "__main__":
    # Limits for LHS
    limits = np.array([h_s0_range, h_s1_range, w_s0_range, w_s1_range])

    # Sample using LHS
    num_samples = 12
    sampling = LHS(xlimits=limits)
    samples = sampling(num_samples)

    # Compute training data
    data_points = []
    for sample in samples:
        h_s0 = sample[0]
        h_s1 = sample[1]
        w_s0 = sample[2]
        w_s1 = sample[3]

        print(f"h_s0 = {h_s0}, h_s1 = {h_s1}, w_s0 = {w_s0}, w_s1 = {w_s1}")

        # Calculate data
        rot_angles, torques = create_model(h_s0, h_s1, w_s0, w_s1)
        print(f"Rotational angles = {rot_angles}")
        print(f"Torque = {torques}")

        data = {
            "rot_angles": list(rot_angles),
            "torques": list(torques),
            "h_s0": h_s0,
            "h_s1": h_s1,
            "w_s0": w_s0,
            "w_s1": w_s1
        }
        print(data)
        data_points.append(data)

    with open(f"data_LHS_12.json", "w", encoding="utf-8") as f:
        json.dump(data_points, f, indent=2)