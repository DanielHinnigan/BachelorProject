from femm import *
import os

from src.permanent_magnet import *
from src.constants import *
from src.coils import coils
from src.materials import add_materials
from src.optimization import optimize
# Open document
openfemm()

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
rotor()

# Build stator
construct_stator()

# Insert magnets
magnet_points = permanent_magnets_v2()

# Rotate rotor such that the d-axis is 
mi_selectgroup(1)
mi_selectgroup(2)
mi_moverotate(0, 0, theta_rot)

# Zoom to fit
mi_zoomnatural()

# Boundaries
mi_makeABC(7, outer_R_stator+5, 0, 0, 0)

# Save file
if os.path.exists("femm/v_shape_test.FEM"):
    os.remove("femm/v_shape_test.FEM")
mi_saveas("femm/v_shape_test.FEM")

# Rotate motor
sim = input("Simulate: Yes (Y) or No (N)? ")

# Optimize
optim = input("Optimize: Yes (Y) or No (N)? ")
if optim == "Y":    
    optimize(all_parameters, to_tune)

# Close
input("Press enter to close ")
closefemm()