import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

from constants import *
from create_samples import create_model

plt.style.use("ggplot")

# Get data
with open("data_random.json", "r") as f:
    data = json.load(f)

# Process data
h_s0 = []
h_s1 = []
w_s0 = []
w_s1 = []
avg_Ts = []
ripple_Ts = []
for i in data:
    h_s0.append(i["h_s0"])
    h_s1.append(i["h_s1"])
    w_s0.append(i["w_s0"])
    w_s1.append(i["w_s1"])

    torques = np.array(i["torques"])
    avg_T = np.mean(torques)
    avg_Ts.append(avg_T)
    ripple_Ts.append((np.max(torques)-np.min(torques))/avg_T)

h_s0 = np.array(h_s0)
h_s1 = np.array(h_s1)
w_s0 = np.array(w_s0)
w_s1 = np.array(w_s1)

X_train = np.vstack((h_s0, h_s1, w_s0, w_s1)).T

# Fit GPs
gp_avg_T = GaussianProcessRegressor()
gp_avg_T.fit(X_train, avg_Ts)

gp_ripple = GaussianProcessRegressor()
gp_ripple.fit(X_train, ripple_Ts)

def seq_update(x):
    global X_train

    # Simulate
    rot_angles, torques = create_model(x[0],x[1],x[2],x[3])
    print("Simulation done. Processing data...")

    # Append data to json file
    with open("data_random.json", "r") as f:
        data = json.load(f)

    new_data = {
        "rot_angles": list(rot_angles),
        "torques": list(torques),
        "h_s0": x[0],
        "h_s1": x[1],
        "w_s0": x[2],
        "w_s1": x[3]
    }

    data.append(new_data)


    with open("data_random.json", "w") as f:
        json.dump(data, f, indent=2)

    # Process data
    torques = np.array(torques)
    avg_T = np.mean(torques)
    ripple_T = (np.max(torques)-np.min(torques))/avg_T

    avg_Ts.append(avg_T)
    ripple_Ts.append(ripple_T)

    # Fit GPs to new data
    X_train = np.vstack((X_train, x))
    gp_avg_T.fit(X_train, avg_Ts)
    gp_ripple.fit(X_train, ripple_Ts)
    print(f"Done processing data: avg_T = {avg_T}, ripple = {ripple_T}. Moving on to next data point")

    # Return (Torque is negative, so also minimize to maximize)
    return avg_T+ripple_T

# Optimize using GPs
def loss(x):
    # Convert into 2D
    print("----------------------")
    print(f"h_s0 = {x[0]}, h_s1 = {x[1]}, w_s0 = {x[2]}, w_s1 = {x[3]}")

    x = [x] # Needed for prediction
    # Perform predictions: Predict using both GPs, if one GP has a large variance, use the FEMM model
    avg_T_pred, err = gp_avg_T.predict(x, return_std=True)
    print(f"Predicted avg_T = {avg_T_pred}, err = {err}")
    if np.abs(err/avg_T_pred) > err_threshold:
        print(f"Error divided by avg_T is {np.abs(err/avg_T_pred)}. Starting simulation...")
        return seq_update(x[0])
    
    ripple_pred, err = gp_ripple.predict(x, return_std=True)
    print(f"Predicted ripple = {ripple_pred}, err = {err}")
    if np.abs(err/ripple_pred) > err_threshold:
        print(f"Error divided by avg_T is {np.abs(err/ripple_pred)}. Starting simulation...")
        return seq_update(x[0])

    print("---------------------------------------------")
    # If variance is low, use the GPs for prediction
    return avg_T_pred+ripple_pred

if __name__ == "__main__":
    for _ in range(10):
        # Random initial conditions
        h_s0_initial = np.random.uniform(h_s0_range[0], h_s0_range[1])
        h_s1_initial = np.random.uniform(h_s1_range[0], h_s1_range[1])
        w_s0_initial = np.random.uniform(w_s0_range[0], w_s0_range[1])
        w_s1_initial = np.random.uniform(w_s1_range[0], w_s1_range[1])

        x_initial = [h_s0_initial, h_s1_initial, w_s0_initial, w_s1_initial]
        x_bounds = [h_s0_range, h_s1_range, w_s0_range, w_s1_range]

        res = minimize(loss, x_initial, bounds=x_bounds)
        print(res)