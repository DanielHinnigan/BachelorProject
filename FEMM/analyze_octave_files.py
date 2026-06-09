from femm import *
import os
import numpy as np

# Folder to analyse
folder = "femm/Rotations"

# Open document
openfemm()

# Get files to analyse
files = os.listdir(folder)

files = [f for f in files if f.endswith(".FEM")]
data_points = []

# Analyse files
for file in files:
    # Get angle w.r.t. d-axis
    angle = int(file[8:-4])

    # Open file
    opendocument(f"{folder}/{file}")

    # Analyse
    mi_analyze(1)
    mi_loadsolution()

    # Calculate Torque
    mo_groupselectblock(1)
    torque = mo_blockintegral(22)
    print(f"Torque is equal to = {torque:.2f} Nm at angle = {angle}")
    mo_clearblock()

    # Flux linkage for no-load
    #flux_linkage_a = mo_getcircuitproperties("A")[2]
    #print(f"Flux Linkage is equal to = {flux_linkage_a} Nm at angle = {angle}")
        
    data_points.append([angle, torque])
    mo_close()
    mi_close()
print(data_points)
# Save as np array
data_points = np.array(data_points)
np.savetxt("datapoints_femm_4poles_75pm.txt", data_points)