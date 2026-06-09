from femm import *
import os
from src.constants import theta_rot
import numpy as np

# Folder to analyse
theta_pms = np.linspace(10, 75, 10)/180*np.pi
for x in theta_pms:
    folder = f"femm/theta_pms/theta_pm{x/np.pi*180:.2f}"

    # Open document
    openfemm()

    # Get files to analyse
    files = os.listdir(folder)

    # Only take files that end on .FEM
    files = [f for f in files if f.endswith(".FEM")]

    # Analyse files
    data_points = []
    rot_angles = np.arange(0, 180, 5)
    for i in range(len(files)):
        # Get theta_pm angle
        rot = rot_angles[i]

        # Open file
        opendocument(f"{folder}/rot{rot+theta_rot}.FEM")

        # Analyse
        mi_analyze(1)
        mi_loadsolution()

        # Calculate Torque
        mo_groupselectblock(1)
        torque = mo_blockintegral(22)
        print(f"Torque is equal to = {torque:.2f} Nm at angle = {rot+theta_rot}")
        mo_clearblock()

        
        # Append datapoint and close windows
        data_points.append([rot+theta_rot, torque])
        mo_close()
        mi_close()

    # Log datapoints if something goes wrong with saving the file
    print(data_points)

    # Save as np array
    data_points = np.array(data_points)
    np.savetxt(f"datapoints_femm_4poles_pm_angle{x}.txt", data_points)