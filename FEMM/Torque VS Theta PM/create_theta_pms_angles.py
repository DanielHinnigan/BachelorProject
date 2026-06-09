from femm import *
import os

from src.permanent_magnet import *
from src.constants import *
import src.constants as constants
from src.coils import coils
from src.materials import add_materials
from src.optimization import optimize

def construct_stator():
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

        coils()

# Build rotor
def rotor():
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

if __name__ == "__main__":
    beta = 0.7
    rot_angle = 5

    # Iterate over theta pms
    theta_pms = np.linspace(10, 75, 10)/180*np.pi
    poles_pairs = num_poles/2
    for x in theta_pms:
        # Open FEMM
        openfemm()

        # Update the theta_pm in the global constants file
        constants.theta_pm = x

        # Create a new document
        newdocument(0)
        main_maximize()
        mi_probdef(0, 'millimeters', 'planar', 1e-8, length, 30)

        # Add materials
        add_materials()

        # Material outside motor
        mi_addblocklabel(outer_R_stator+10, 0)
        mi_selectlabel(outer_R_stator+10, 0)
        mi_setblockprop("Air")
        mi_clearselected()

        # Build rotor
        rotor()

        # Build stator
        construct_stator()

        # Insert magnets
        magnet_points = permanent_magnets_v2()
        
        # Zoom to fit
        mi_zoomnatural()

        # Boundaries
        mi_makeABC(7, None, 0, 0, 0)

        # Check if folder exists
        path = f"femm/theta_pms/theta_pm{x/np.pi*180:.2f}/"
        if not os.path.exists(path):
            os.mkdir(path)

        # Rotate to align with the d-axis
        mi_selectgroup(1)
        mi_selectgroup(2)
        mi_moverotate(0,0,theta_rot)
        
        # Iterate over rotational angles
        for rot in np.arange(0, 180, rot_angle):
            # Rotate rotor such that the d-axis is aligned with phase A
            if rot > 0:
                mi_selectgroup(1)
                mi_selectgroup(2)
                mi_moverotate(0, 0, -rot_angle)

            # Change currents
            I_a = I_d*np.cos(np.radians(poles_pairs*rot))-I_q*np.sin(np.radians(poles_pairs*rot))
            I_b = I_d*np.cos(np.radians(poles_pairs*rot-120))-I_q*np.sin(np.radians(poles_pairs*rot-120))
            I_c = I_d*np.cos(np.radians(poles_pairs*rot+120))-I_q*np.sin(np.radians(poles_pairs*rot+120))

            mi_modifycircprop('A', 1, I_a)
            mi_modifycircprop('B', 1, I_b)
            mi_modifycircprop('C', 1, I_c)

            # Save file
            file = f"rot{theta_rot+rot}.FEM"

            if os.path.exists(path+file):
                 os.remove(path+file)
            mi_saveas(path+file)

        # Close femm
        closefemm()