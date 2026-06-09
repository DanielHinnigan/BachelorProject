import os
import numpy as np
import json

# Import files
folder = "Data - NumPy/"
files = os.listdir(folder)

# Open and process data in files
theta_pms = np.linspace(10, 75, 10)/180*np.pi
metrics = []
for i in range(len(files)):
    # Find file
    file = files[i]
    data = np.loadtxt(folder + file)
    
    # Process
    rot_angles = data[:,0]
    torques = np.abs(data[:,1])

    avg_T = np.mean(torques)
    ripple_T = np.mean(np.abs(torques-avg_T))/avg_T
    
    # Save data
    theta_pm = theta_pms[i]
    point = {
        "theta_pm": theta_pm,
        "avg_T": avg_T,
        "ripple_T": ripple_T,
        "torques": list(torques),
        "rotational angles": list(rot_angles) 
    }
    metrics.append(point)
    print(f"Theta pm = {theta_pm*180/np.pi:.2f} degrees, Average Torque = {avg_T:.2f} Nm, Ripple Torque = {ripple_T*100.0:.2f} %")
# --------------------------------------
with open("post_processed.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)