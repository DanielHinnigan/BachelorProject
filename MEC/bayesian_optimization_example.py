"""
This code performs optimization on the model from JeihoonV7 using Bayesian Optimization
"""

"""
This script is used to implement torque vs magnetic length, theta_pm, R_mi

Code uses a new circuit for the D-axis which is inspired by the flow of flux lines when the circuit
is exited by the d-axis sources. This MEC is also inspired from the "inductance calculation" paper.

It was observed that the permeabilities of the various reluctances fluctated between two numbers
when using the calculated B-field as the new B-field. A "softer" approach is used to iteratively update the B-fields which resulted in a convergence of the B-fields
and permeabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import pyswarms as ps

from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.preprocessing import StandardScaler

from smt.sampling_methods import LHS

from materials import *
import constants
from reluctances import *
from permanent_magnet import *

verbose = 2
alpha = 0.1
tol = 1e-3

def solve_pm_equations(iter = 100):
    # Initial conditions: As a start, assume satured iron
    B_y1 = np.random.uniform(0.1,2)
    B_y2 = np.random.uniform(0.1,2)
    B_y3 = np.random.uniform(0.1,2)
    B_b = np.random.uniform(0.1,2)

    # Linear reluctance
    R_g = r_g_d_v6()
    R_m = constants.l_c/constants.A_c/constants.mu0
    R_a = r_a_d_v6()

    # ROUGH MEASSUREMENTS
    phi_d=0

    # Flux sources
    phi_m = constants.B_r*constants.A_m

    # Parameters for convergence
    B_old = np.array([float("inf"), float("inf"), float("inf"), float("inf")])

    x = 0
    # ------------------------ CONVERGE ON B-FIELD ----------------------------------------
    for _ in range(iter):
        # Update permeabilities
        mu_y1 = B_y1/interpolate_H(B_y1)
        mu_y2 = B_y2/interpolate_H(B_y2)
        mu_y3 = B_y3/interpolate_H(B_y3)
        mu_b = B_b/interpolate_H(B_b)

        # Nonlinear reluctances
        R_y1, A_y1 = r_y1_d_v6(mu_y1)
        R_y2, A_y2 = r_y2_d_v6(mu_y2)
        R_y3, A_y3 = r_y3_d_v6(mu_y3)
        R_b, A_b = r_r_v3(mu_b)

        # Nodes order: A|B|C|D|E|F|G|H|I|J|K|Z
        # Potential order: The same
        # PM flux linkage is calculted under the no-load condition (I_d = I_q = 0 -> F_ds1 = 0)
        b = np.array([0, 0, -phi_m, 0, 0, 0, phi_m, 0, 0, phi_m, -phi_m, 0])
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A
            [-1/R_g, 1/R_g+1/R_b+1/R_y1, -1/R_y1, 0, 0, 0, 0, 0, -1/R_b, 0, 0, 0], # B
            [0, -1/R_y1, 1/R_y1+1/R_m+1/R_a, -1/R_a, 0, 0, 0, 0, 0, -1/R_m, 0,0], #C
            [0, 0, -1/R_a, 1/R_a+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0, 0], #D
            [0, 0, 0, -1/R_y3, 1/R_y3+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0], #E
            [0, 0, 0, 0, -1/R_y3, 1/R_y3+1/R_a, -1/R_a, 0, 0, 0, 0, 0], #F
            [0, 0, 0, 0, 0, -1/R_a, 1/R_a+1/R_m+1/R_y1, -1/R_y1, 0, 0, -1/R_m, 0], #G
            [0, 0, 0, 0, 0, 0, -1/R_y1, 1/R_y1+1/R_g+1/R_b, -1/R_b, 0,0,-1/R_g], #H
            [0, -1/R_b, 0, 0, 0, 0, 0, -1/R_b, 2/R_b,0,0,0], #I
            [0, 0, -1/R_m, 0, 0, 0, 0, 0, 0, 1/R_m+1/R_y2, -1/R_y2, 0], #J
            [0, 0, 0, 0, 0, 0, -1/R_m, 0, 0, -1/R_y2, 1/R_y2+1/R_m, 0], # K
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # Z
        ])

        try:
            x = np.linalg.solve(A, b)

            # Update flux densities
            B_y1_calc = np.abs(x[1]-x[2])/A_y1/R_y1
            B_y1 -= alpha*(B_y1-B_y1_calc)

            B_y2_calc = np.abs(x[-3]-x[-2])/A_y2/R_y2
            B_y2 -= alpha*(B_y2-B_y2_calc)

            B_y3_calc = np.abs(x[3]-x[4])/A_y3/R_y3
            B_y3 -= alpha*(B_y3-B_y3_calc)

            B_b_calc = np.abs(x[1]-x[-4])/A_b/R_b
            B_b -= alpha*(B_b-B_b_calc)

            # Check for convergence
            B = np.array([B_y1, B_y2, B_y3, B_b])
            if np.linalg.norm(B-B_old)/np.linalg.norm(B) < tol:
                break
            B_old = B

        except np.linalg.LinAlgError as e:
            print("----------------------------------------------")
            print(f"Matrix is singular in D-axis: {e}")
            """print(f"Matrix is: {A}")
            print(f"Vector is {b}")"""
            print("Reset B-fields")
            print("----------------------------------------------")
            B_y1 = np.random.uniform(0.1,2)
            B_y2 = np.random.uniform(0.1,2)
            B_y3 = np.random.uniform(0.1,2)
            B_b = np.random.uniform(0.1,2)

    # ------------------------------- D-axis flux due to PM -----------------------------
    phi_pm = (x[-5]-x[-1])/R_g

    return phi_pm

def calculate_L_d(I_d, iter = 100):
    # Initial conditions: As a start, assume satured iron
    B_y1 = np.random.uniform(0.1,2)
    B_y2 = np.random.uniform(0.1,2)
    B_y3 = np.random.uniform(0.1,2)
    B_b = np.random.uniform(0.1,2)

    # Linear reluctance
    R_g = r_g_d_v6()
    R_m = constants.l_c/constants.A_c/constants.mu0
    R_a = r_a_d_v6()

    # ROUGH MEASSUREMENTS
    phi_d=0
    F_ds = I_d*constants.N_c

    # Flux sources
    phi_m = constants.B_r*constants.A_m

    x = 0

    # Parameters for convergence
    B_old = np.array([float("inf"), float("inf"), float("inf"), float("inf")])

    # ------------------------ CONVERGE ON B-FIELD ----------------------------------------
    for _ in range(iter):
        # Update permeabilities
        mu_y1 = B_y1/interpolate_H(B_y1)
        mu_y2 = B_y2/interpolate_H(B_y2)
        mu_y3 = B_y3/interpolate_H(B_y3)
        mu_b = B_b/interpolate_H(B_b)

        # Nonlinear reluctances
        R_y1, A_y1 = r_y1_d_v6(mu_y1)
        R_y2, A_y2 = r_y2_d_v6(mu_y2)
        R_y3, A_y3 = r_y3_d_v6(mu_y3)
        R_b, A_b = r_r_v3(mu_b)

        # Nodes order: A|B|C|D|E|F|G|H|I|J|K|Z
        # Potential order: The same
        # PM flux linkage is calculted under the no-load condition (I_d = I_q = 0 -> F_ds1 = 0)
        b = np.array([F_ds, 0, -phi_m, 0, 0, 0, phi_m, 0, 0, phi_m, -phi_m, 0])
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A
            [-1/R_g, 1/R_g+1/R_b+1/R_y1, -1/R_y1, 0, 0, 0, 0, 0, -1/R_b, 0, 0, 0], # B
            [0, -1/R_y1, 1/R_y1+1/R_m+1/R_a, -1/R_a, 0, 0, 0, 0, 0, -1/R_m, 0,0], #C
            [0, 0, -1/R_a, 1/R_a+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0, 0], #D
            [0, 0, 0, -1/R_y3, 1/R_y3+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0], #E
            [0, 0, 0, 0, -1/R_y3, 1/R_y3+1/R_a, -1/R_a, 0, 0, 0, 0, 0], #F
            [0, 0, 0, 0, 0, -1/R_a, 1/R_a+1/R_m+1/R_y1, -1/R_y1, 0, 0, -1/R_m, 0], #G
            [0, 0, 0, 0, 0, 0, -1/R_y1, 1/R_y1+1/R_g+1/R_b, -1/R_b, 0,0,-1/R_g], #H
            [0, -1/R_b, 0, 0, 0, 0, 0, -1/R_b, 2/R_b,0,0,0], #I
            [0, 0, -1/R_m, 0, 0, 0, 0, 0, 0, 1/R_m+1/R_y2, -1/R_y2, 0], #J
            [0, 0, 0, 0, 0, 0, -1/R_m, 0, 0, -1/R_y2, 1/R_y2+1/R_m, 0], # K
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # Z
        ])

        try:
            x = np.linalg.solve(A, b)

            # Update flux densities
            B_y1_calc = np.abs(x[1]-x[2])/A_y1/R_y1
            B_y1 -= alpha*(B_y1-B_y1_calc)

            B_y2_calc = np.abs(x[-3]-x[-2])/A_y2/R_y2
            B_y2 -= alpha*(B_y2-B_y2_calc)

            B_y3_calc = np.abs(x[3]-x[4])/A_y3/R_y3
            B_y3 -= alpha*(B_y3-B_y3_calc)

            B_b_calc = np.abs(x[1]-x[-4])/A_b/R_b
            B_b -= alpha*(B_b-B_b_calc)

            # Check for convergence
            B = np.array([B_y1, B_y2, B_y3, B_b])
            if np.linalg.norm(B-B_old)/np.linalg.norm(B) < tol:
                break
            B_old = B
        except np.linalg.LinAlgError as e:
            print("----------------------------------------------")
            print(f"Matrix is singular in D-axis: {e}")
            """print(f"Matrix is: {A}")
            print(f"Vector is {b}")"""
            print("Reset B-fields")
            print("----------------------------------------------")
            B_y1 = np.random.uniform(0.1,2)
            B_y2 = np.random.uniform(0.1,2)
            B_y3 = np.random.uniform(0.1,2)
            B_b = np.random.uniform(0.1,2)

    # ------------------------------- D-axis flux due to D-axis source -----------------------------
    # Solve the circuit equations again using only the D-axis source.
    # The reluctances calculated using the iterative approach (both d-axis sources included) are used in the circuit
    b = np.array([F_ds, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A
        [-1/R_g, 1/R_g+1/R_b+1/R_y1, -1/R_y1, 0, 0, 0, 0, 0, -1/R_b, 0, 0, 0], # B
        [0, -1/R_y1, 1/R_y1+1/R_m+1/R_a, -1/R_a, 0, 0, 0, 0, 0, -1/R_m, 0,0], #C
        [0, 0, -1/R_a, 1/R_a+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0, 0], #D
        [0, 0, 0, -1/R_y3, 1/R_y3+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0, 0], #E
        [0, 0, 0, 0, -1/R_y3, 1/R_y3+1/R_a, -1/R_a, 0, 0, 0, 0, 0], #F
        [0, 0, 0, 0, 0, -1/R_a, 1/R_a+1/R_m+1/R_y1, -1/R_y1, 0, 0, -1/R_m, 0], #G
        [0, 0, 0, 0, 0, 0, -1/R_y1, 1/R_y1+1/R_g+1/R_b, -1/R_b, 0,0,-1/R_g], #H
        [0, -1/R_b, 0, 0, 0, 0, 0, -1/R_b, 2/R_b,0,0,0], #I
        [0, 0, -1/R_m, 0, 0, 0, 0, 0, 0, 1/R_m+1/R_y2, -1/R_y2, 0], #J
        [0, 0, 0, 0, 0, 0, -1/R_m, 0, 0, -1/R_y2, 1/R_y2+1/R_m, 0], # K
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # I
    ])
    x = np.linalg.solve(A, b)
    phi_d = (x[-5]-x[-1])/R_g
    lambda_d_source = constants.N_c*constants.k_w*phi_d

    # Inductance d-axis
    L_d = lambda_d_source/I_d

    return L_d

def calculate_L_q(I_q, iter = 1000):
    # Initial conditions: As a start, assume satured iron
    B_y1 = 1
    B_y2 = 1
    B_y3 = 1
    B_r = 1

    # Test current
    F_q1 = I_q*constants.N_c

    phi_q=0
    L_q = 0

    # Linear reluctances
    R_g1 = r_g1_y_axis()
    R_g2 = r_g2()
    
    x = 0

    # Parameters for convergence
    B_old = np.array([float("inf"), float("inf"), float("inf"), float("inf")])

    # Iterative algorithm
    for i in range(iter):
        # Permeability
        mu_y1 = B_y1/interpolate_H(B_y1)
        mu_y2 = B_y2/interpolate_H(B_y2)
        mu_y3 = B_y3/interpolate_H(B_y3)
        mu_r = B_r/interpolate_H(B_r)

        # Nonlinear reluctances
        R_y1, A_y1 = r_y1(mu_y1)
        R_y2, A_y2 = r_y2(mu_y2)
        R_y3, A_y3 = r_y3_d_v6(mu_y3)
        R_r, A_r = r_r_v3(mu_r)

        # Nodes order: A | B | C | D | E | F | G | H | I
        # Potential order: The same
        b = np.array([F_q1, 0, 0, 0, 0, 0, 0, 0, 0])
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0], # A
            [-1/R_g2, 1/R_g2+1/R_r+1/R_y2, -1/R_y2, 0, 0, 0, -1/R_r, 0, 0], # B
            [0, -1/R_y2, 1/R_y2+1/R_y3, -1/R_y3, 0, 0, 0, 0, 0], # C
            [0, 0, -1/R_y3, 1/R_y3+1/R_y2, -1/R_y2, 0, 0, 0, 0], # D
            [0, 0, 0, -1/R_y2, 1/R_y2+1/R_r+1/R_g2, 0, 0, -1/R_r, -1/R_g2], # E
            [-1/R_g1, 0, 0, 0, 0, 2/R_g2, 0, 0, -1/R_g1], # F
            [0, -1/R_r, 0, 0, 0, 0, 1/R_r+1/R_y1, -1/R_y1, 0], # G
            [0, 0, 0, 0, -1/R_r, 0, -1/R_y1, 1/R_y1+1/R_r, 0], # H
            [0, 0, 0, 0, 0, 0, 0, 0, 1], # I
        ])

        x = np.linalg.solve(A, b)

        # Update flux densities
        B_r_calc = np.abs(x[1]-x[-3])/A_r/R_r
        B_r -= alpha*(B_r-B_r_calc)
        
        B_y1_calc = np.abs(x[-3]-x[-2])/A_y1/R_y1
        B_y1 -= alpha*(B_y1-B_y1_calc)

        B_y2_calc = np.abs(x[1]-x[2])/A_y2/R_y2
        B_y2 -= alpha*(B_y2-B_y2_calc)

        B_y3_calc = np.abs(x[2]-x[3])/A_y3/R_y3
        B_y3 -= alpha*(B_y3-B_y3_calc)

        # Check for convergence
        B = np.array([B_r, B_y1, B_y2, B_y3])
        if np.linalg.norm(B-B_old)/np.linalg.norm(B) < tol:
            break
        B_old = B

    # Calculate flux
    phi_q1 = x[0]/R_g1
    phi_q2 = x[4]/R_g2

    phi_q = phi_q1+phi_q2

    # Flux linkage
    lambda_q = constants.N_c*constants.k_w*phi_q

    # Inductance d-axis
    L_q = lambda_q/I_q

    return L_q


def create_samples():
    # Create samples using LHS
    xlimits = np.array([[0.0, 10.]])

    num_samples = 3
    sampling = LHS(xlimits=xlimits)
    samples = sampling(num_samples)

    f = objective_func_BO(samples)
    return samples, f

def fit_gp(X, f):
    kernel = (
        ConstantKernel(constant_value=1.0) 
        * RBF(length_scale=1.0, length_scale_bounds = (1e-3, 1e2))
    )

    """kernel = (
        ConstantKernel(constant_value=1.0) 
        * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
    )"""

    gp = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=200,
        #alpha=0.01,
        random_state=1 # For reproducability
    ).fit(X, f)
    print(gp.kernel_)
    print(gp.log_marginal_likelihood())
    return gp

def EI(xs, gp, f_opt):
    # Predict on GP
    mu_x, sigma_x = gp.predict(xs, return_std=True)
    mu_x = mu_x[0]
    sigma_x = sigma_x[0]

    # Helper values
    inc = mu_x-f_opt

    # Calculate ei
    ei = inc*norm.cdf(inc/sigma_x)+sigma_x*norm.pdf(inc/sigma_x)
    #ei = np.max((0., inc))+sigma_x*norm.pdf(inc/sigma_x)-np.abs(inc)*norm.cdf(inc/sigma_x)

    return ei

def argmax_ei(gp, f_opt, X_scaler:StandardScaler):

    """# Hyper parameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, bounds=([0.0], [10.]), options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(lambda x: -EI(x, gp, f_opt), iters=1000, verbose =True)"""

    num_initials = 10
    f_best = -np.inf
    x_best = None

    lb = X_scaler.transform([[0.0]])[0,0]
    ub = X_scaler.transform([[10.0]])[0,0]
    xlimits = np.array([[lb, ub]])

    num_initials = 10
    sampling = LHS(xlimits=xlimits)
    samples = sampling(num_initials)

    for i in range(num_initials):
        res = minimize(
            lambda x: -EI([x], gp, f_opt), 
            samples[i], 
            bounds=xlimits
        )

        if f_best < -res["fun"]:
            f_best = -res["fun"]
            x_best = res["x"]

    return x_best[0], f_best

def objective_func_BO(x):
    return x*np.sin(x)

if __name__ == "__main__":
    x_bounds = [0.0, 10.]
    max_iter = 5
    patience = 10
    idx = 0
    history = []
    cum_time = 0

    X, f = create_samples()
    print(X)
    print(f)
    for _ in range(max_iter):
        start_time = time.perf_counter()

        # Scale data to unit variance and zero mean
        X_scaler = StandardScaler()
        f_scaler = StandardScaler()

        X_scaler.fit(X)
        f_scaler.fit(f)

        X_scaled = X_scaler.transform(X)
        f_scaled = f_scaler.transform(f)

        print(f"X_scaled = {X_scaled}")
        print(f"f_scaled = {f_scaled}")

        f_opt = np.max(f_scaled)

        # Fit model
        gp = fit_gp(X_scaled, f_scaled)

        # Find new point
        x_n, ei_max = argmax_ei(gp, f_opt, X_scaler)

        # Plot
        fig, axs = plt.subplots(1, 2)
        
        x_plot = X_scaler.transform(np.reshape(np.linspace(x_bounds[0], x_bounds[1], 1000), shape=(-1, 1)))
        mu_plot, sigma_plot = gp.predict(x_plot, return_std=True)
        
        axs[0].plot(x_plot, mu_plot, label="Mean Prediction")
        axs[0].plot(x_plot, f_scaler.transform(objective_func_BO(X_scaler.inverse_transform(x_plot))), label="True function")
        axs[0].scatter(X_scaled, f_scaled, label="Observations", s=50)
        axs[0].fill_between(
            x_plot.flatten(),
            mu_plot-1.96*sigma_plot,
            mu_plot+1.96*sigma_plot,
            alpha=0.5,
            label="95% confidence interval",
            color="tab:orange"
        )
        axs[0].set_xlabel('$x$')
        axs[0].set_ylabel('$f(x)$')
        axs[0].set_title('Gaussian process regression on MEC')
        axs[0].legend(loc='upper right')

        axs[1].plot(x_plot, [EI([x], gp, f_opt) for x in x_plot], label="EI")
        axs[1].scatter(X_scaled, [EI([x], gp, f_opt) for x in X_scaled], label="Observed Points", s=50)
        axs[1].scatter(x_n, ei_max, label="Maximum EI", s=75)
        axs[1].legend()
        axs[1].set_xlabel('$x$')
        axs[1].set_ylabel('$EI(x)$')
        axs[1].set_title('Expected Improvement on MEC')
        axs[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()

        # For logging
        mu_x, sigma_x = gp.predict([[x_n]], return_std=True)

        # Scale next sample into correct using  by destandizing
        x_n = X_scaler.inverse_transform([[x_n]])

        # Sample objective function and append
        f_n = objective_func_BO(x_n)

        # Update data
        X = np.vstack((X, x_n))
        f = np.vstack((f, f_n))

        print(f"X = \n{X}")
        print(f"f = \n{f}")

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        cum_time += total_duration

        # Log
        f_opt = f_scaler.inverse_transform([[f_opt]])
        print(f"Current iter: {_}; Current sample (rescaled) = {x_n}, output = {f_n}, current maximum = {f_opt}, maximum EI = {ei_max}, cum_time = {cum_time}, var = {sigma_x**2}, mean = {mu_x}")

        # Append to history
        history.append(
            {
                "Sample": x_n,
                "Output": f_n,
                "Current maximum": f_opt,
                "Maximum EI": ei_max,
                "Predictive Variance": sigma_x**2,
                "Predictive mean": mu_x,
                "Cumulative computation time": cum_time
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