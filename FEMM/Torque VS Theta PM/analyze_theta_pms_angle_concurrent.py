from femm import *
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.constants import theta_rot

def analyse(rot, folder):
    # Open document
    openfemm()

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

    
    # Close window
    mo_close()
    mi_close()

    return rot+theta_rot, torque

# Folder to analyse
if __name__ == "__main__":
    theta_pms = np.linspace(10, 75, 10)/180*np.pi
    for x in theta_pms:
        folder = f"femm/theta_pms/theta_pm{x/np.pi*180:.2f}"

        # Analyse files
        data_points = []
        max_workers = 6 # Number of cpu cores to use
        rot_angles = np.arange(0, 180, 5) # Taken from create_theta_pms_angles.py
        print(f"Starting parallel simulation for {len(rot_angles)} angles with {max_workers} workers.")

        # Create a partial with the folder fixed
        analyse_with_folder = partial(analyse, folder=folder)

        # Run workers
        with ProcessPoolExecutor(max_workers=max_workers) as e:
            results = e.map(analyse_with_folder, rot_angles)
        results = np.array(list(results))
        print(results)

        # Save as np array
        np.savetxt(f"datapoints_femm_4poles_pm_angle{x/np.pi*180:.2f}.txt", data_points)