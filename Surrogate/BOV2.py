"""
BO with different length scales for each input feature
"""

import numpy as np
import json
import time

from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern

from sklearn.preprocessing import StandardScaler

from smt.sampling_methods import LHS

from constants import h_s0_range, h_s1_range, w_s0_range, w_s1_range
from create_samples import create_model

def EI(x, gp, f_opt):

    # Predict on GP
    mu_x, sigma_x = gp.predict(x, return_std=True)
    mu_x = mu_x[0]
    sigma_x = sigma_x[0]

    # Helper values
    inc = mu_x-f_opt

    # Calculate ei
    ei = inc*norm.cdf(inc/sigma_x)+sigma_x*norm.pdf(inc/sigma_x)
    #ei = np.max((0., inc))+sigma_x*norm.pdf(inc/sigma_x)-np.abs(inc)*norm.cdf(inc/sigma_x)

    return ei

def fit_gp(X, f): 
    kernel = (
        ConstantKernel() 
        #* RBF(length_scale=[1.0]*4)
        * Matern(length_scale = [1.0]*4)
    )

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100).fit(X, f)
    print(gp.kernel_)
    print(gp.log_marginal_likelihood())
    return gp

def argmax_ei(gp, f_opt):
    limits = np.array([h_s0_range, h_s1_range, w_s0_range, w_s1_range])

    num_initials = 50
    sampling = LHS(xlimits=limits)
    samples = sampling(num_initials)

    f_best = -np.inf
    x_best = None

    for i in range(num_initials):
        res = minimize(
            lambda x: -EI([x], gp, f_opt), 
            samples[i], 
            bounds=limits
        )

        if f_best < -res["fun"]:
            f_best = -res["fun"]
            x_best = res["x"]
    print(x_best)
    return x_best, f_best

def objective_func_BO(x):
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

    # Return (Torque is negative, so also minimize to maximize)
    # This BO formulation is a maximization problem, so maximize the negative of the torque (which is positive) and minimize the torque ripple
    return -(avg_T+ripple_T)

def import_data():
    # Get data
    with open("data_LHS_12.json", "r") as f:
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

    # Convert to objective function
    f = -(np.array(avg_Ts)+np.array(ripple_Ts))
    
    f_opt = np.max(f)

    # Return
    return X_train, f, f_opt

if __name__ == "__main__":
    X, f, f_opt = import_data()
    print(X)
    print(f)
    print(f_opt)
    max_iter = 100
    patience = 10
    idx = 0
    history = []
    cum_time = 0
    for _ in range(max_iter):
        start_time = time.perf_counter()

        # Fit model
        gp = fit_gp(X, f)

        # Find new point
        x_n, ei_max = argmax_ei(gp, f_opt)

        # For logging
        mu_x, sigma_x = gp.predict([x_n], return_std=True)

        # Sample objective function and append
        f_n = objective_func_BO(x_n)

        # Update data
        X = np.vstack((X, x_n))
        f = np.hstack((f, f_n))

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        cum_time += total_duration

        # Log
        print(f"Current iter: {_}; Current sample = {x_n}, output = {f_n}, current maximum = {f_opt}, maximum EI = {ei_max}, cum_time = {cum_time}, var = {sigma_x**2}, mean = {mu_x}")
        
        history.append(
            {
                "Sample": x_n,
                "Output": f_n,
                "Current maximum": f_opt,
                "Maximum EI": ei_max,
                "Predictive Variance": sigma_x**2,
                "Predictive mean": mu_x,
                "Cumulative computation time": cum_time,
                "Kernel": gp.kernel_,
                "Log-likelihood": gp.log_marginal_likelihood()
            }
        )

        if f_n > f_opt:
            f_opt = f_n
            idx = 0
        else:
            idx += 1

        # Check for early stopping: No improvement last 10 iterations
        if idx == patience:
            print(f"No improvement for the last {patience} iterations. Stopping optimization...")
            break

    print(history)