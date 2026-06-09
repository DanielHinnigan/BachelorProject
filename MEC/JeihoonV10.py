"""
Based on V7, but with the Q-axis circuit changed to include the saturation effect of R_g1 on R_y1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker

from scipy.optimize import minimize, OptimizeResult

from materials import *
import constants
from reluctances import *
from permanent_magnet import *

from constraints import magnet_length_constraint_func,theta_pm_constraint_func

"""
Verbose:
    =3: Print B-fields, fluxes, reluctances of each circuit after convergence
    >=1: Print L_D, L_Q, Lambda_PM, F_Q, F_D, I_D, I_Q
"""
verbose = 3
alpha = 0.1
tol = 1e-3

# Normalization constants
theta_pm_max = 35*np.pi/180
R_A_max = 0.055
w_air_max = 0.4

plt.style.use('ggplot')

def solve_pm_equations(iter = 100):
    # Initial conditions: As a start, assume satured iron
    B_y1 = np.random.uniform(0.1,2)
    B_y2 = np.random.uniform(0.1,2)
    B_y3 = np.random.uniform(0.1,2)
    B_b = np.random.uniform(0.1,2)

    # Linear reluctance
    R_g, A_g = r_g_d_v6()
    R_m = constants.l_c/constants.A_c/constants.mu0
    R_a, A_a = r_a_d_v6()

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
        mu_y1 = np.abs(B_y1)/interpolate_H(np.abs(B_y1))
        mu_y2 = np.abs(B_y2)/interpolate_H(np.abs(B_y2))
        mu_y3 = np.abs(B_y3)/interpolate_H(np.abs(B_y3))
        mu_b = np.abs(B_b)/interpolate_H(np.abs(B_b))

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
            B_y1_calc = (x[1]-x[2])/A_y1/R_y1
            B_y1 -= alpha*(B_y1-B_y1_calc)

            B_y2_calc = (x[-3]-x[-2])/A_y2/R_y2
            B_y2 -= alpha*(B_y2-B_y2_calc)

            B_y3_calc = (x[3]-x[4])/A_y3/R_y3
            B_y3 -= alpha*(B_y3-B_y3_calc)

            B_b_calc = (x[1]-x[-4])/A_b/R_b
            B_b -= alpha*(B_b-B_b_calc)

            # Check for convergence: Upon convergence, B_b is practically equalt to B_b_calc as the change in B_b is practically zero.
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

    # Log
    if verbose == 3:
        print("--------------------- D-axis Magnet ---------------------")
        print(f"B_b = {B_b}")
        print(f"B_y1 = {B_y1}")
        print(f"B_y2 = {B_y2}")
        print(f"B_y3 = {B_y3}")
        print(f"B_g = {(x[0]-x[1])/R_g/A_g}")
        print(f"B_a = {(x[2]-x[3])/R_a/A_a}")
        print(f"B_m = {(x[1]-x[-4])/R_m/constants.A_c}")
        print("")
        print(f"phi_b = {(x[1]-x[-4])/R_b}")
        print(f"phi_y1 = {(x[1]-x[2])/R_y1}")
        print(f"phi_y2 = {(x[-3]-x[-2])/R_y2}")
        print(f"phi_y3 = {(x[3]-x[4])/R_y3}")
        print(f"phi_g = {(x[0]-x[1])/R_g}")
        print(f"phi_a = {(x[2]-x[3])/R_a}")
        print(f"phi_m = {(x[1]-x[-4])/R_m}")

    return phi_pm

def calculate_L_d(I_d, iter = 100):
    # Initial conditions: As a start, assume satured iron
    B_y1 = np.random.uniform(0.1,2)
    B_y2 = np.random.uniform(0.1,2)
    B_y3 = np.random.uniform(0.1,2)
    B_b = np.random.uniform(0.1,2)

    # Linear reluctance
    R_g, A_g = r_g_d_v6()
    R_m = constants.l_c/constants.A_c/constants.mu0
    R_a, A_a = r_a_d_v6()

    # ROUGH MEASSUREMENTS
    phi_d=0
    F_ds = I_d*constants.N_c*2.6184

    # Flux sources
    phi_m = constants.B_r*constants.A_m

    x = 0

    # Parameters for convergence
    B_old = np.array([float("inf"), float("inf"), float("inf"), float("inf")])

    # ------------------------ CONVERGE ON B-FIELD ----------------------------------------
    for _ in range(iter):
        # Update permeabilities
        mu_y1 = np.abs(B_y1)/interpolate_H(np.abs(B_y1))
        mu_y2 = np.abs(B_y2)/interpolate_H(np.abs(B_y2))
        mu_y3 = np.abs(B_y3)/interpolate_H(np.abs(B_y3))
        mu_b = np.abs(B_b)/interpolate_H(np.abs(B_b))

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
            B_y1_calc = (x[1]-x[2])/A_y1/R_y1
            B_y1 -= alpha*(B_y1-B_y1_calc)

            B_y2_calc = (x[-3]-x[-2])/A_y2/R_y2
            B_y2 -= alpha*(B_y2-B_y2_calc)

            B_y3_calc = (x[3]-x[4])/A_y3/R_y3
            B_y3 -= alpha*(B_y3-B_y3_calc)

            B_b_calc = (x[1]-x[-4])/A_b/R_b
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

    # Log
    if verbose == 3:
        print("--------------------- D-axis Inductace ---------------------")
        print(f"B_b = {B_b}")
        print(f"B_y1 = {B_y1}")
        print(f"B_y2 = {B_y2}")
        print(f"B_y3 = {B_y3}")
        print(f"B_g = {(x[0]-x[1])/R_g/A_g}")
        print(f"B_a = {(x[2]-x[3])/R_a/A_a}")
        print(f"B_m = {(x[1]-x[-4])/R_m/constants.A_c}")
        print("")
        print(f"phi_b = {(x[1]-x[-4])/R_b}")
        print(f"phi_y1 = {(x[1]-x[2])/R_y1}")
        print(f"phi_y2 = {(x[-3]-x[-2])/R_y2}")
        print(f"phi_y3 = {(x[3]-x[4])/R_y3}")
        print(f"phi_g = {(x[0]-x[1])/R_g}")
        print(f"phi_a = {(x[2]-x[3])/R_a}")
        print(f"phi_m = {(x[1]-x[-4])/R_m}")
        print("")
        print(f"R_b = {R_b}")
        print(f"R_y1 = {R_y1}")
        print(f"R_y2 = {R_y2}")
        print(f"R_y3 = {R_y3}")
        print(f"R_g = {R_g}")
        print(f"R_a = {R_a}")
        print(f"R_m = {R_m}")
        print("")
        print(f"phi_m = {phi_m}")

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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # Z
    ])
    x = np.linalg.solve(A, b)
    phi_d = (x[-5]-x[-1])/R_g
    lambda_d_source = constants.N_c*constants.k_w*phi_d

    if verbose == 3:
        print(f"phi_g due to d-source: {phi_d}")

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
    R_g1, A_g1 = r_g1_y_axis()
    R_g2, A_g2 = r_g2()
    
    x = 0

    # Parameters for convergence
    B_old = np.array([float("inf"), float("inf"), float("inf"), float("inf")])

    # Iterative algorithm
    for i in range(iter):
        # Permeability
        mu_y1 = np.abs(B_y1)/interpolate_H(np.abs(B_y1))
        mu_y2 = np.abs(B_y2)/interpolate_H(np.abs(B_y2))
        mu_y3 = np.abs(B_y3)/interpolate_H(np.abs(B_y3))
        mu_r = np.abs(B_r)/interpolate_H(np.abs(B_r))

        # Nonlinear reluctances
        R_y1, A_y1 = r_y2(mu_y1) # y2 function is between magnet
        R_y2, A_y2 = r_y1(mu_y2) # y1 function is behind magnet
        R_y3, A_y3 = r_y3_d_v6(mu_y3)
        R_r, A_r = r_r_v3(mu_r)

        # Nodes order: A | B | C | D | E | F | G | H
        # Potential order: The same
        b = np.array([F_q1, 0, 0, 0, 0, 0, 0, 0])
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], # A
            [-1/R_g2, 1/R_g2+1/R_r+1/R_y2, -1/R_y2, 0, 0, -1/R_r, 0, 0], # B
            [0, -1/R_y2, 1/R_y2+1/R_y3, -1/R_y3, 0, 0, 0, 0], # C
            [0, 0, -1/R_y3, 1/R_y3+1/R_y2, -1/R_y2, 0, 0, 0], # D
            [0, 0, 0, -1/R_y2, 1/R_y2+1/R_r+1/R_g2, 0, -1/R_r, -1/R_g2], # E
            [-1/R_g1, -1/R_r, 0, 0, 0, 1/R_g1+1/R_y1+1/R_r, -1/R_y1, 0], # F
            [0, 0, 0, 0, -1/R_r, -1/R_y1, 1/R_r+1/R_y1+1/R_g1, -1/R_g1], # G
            [0, 0, 0, 0, 0, 0, 0, 1], # H
        ])

        x = np.linalg.solve(A, b)

        # Update flux densities
        B_r_calc = (x[1]-x[-3])/A_r/R_r
        B_r -= alpha*(B_r-B_r_calc)
        
        B_y1_calc = (x[-3]-x[-2])/A_y1/R_y1
        B_y1 -= alpha*(B_y1-B_y1_calc)

        B_y2_calc = (x[1]-x[2])/A_y2/R_y2
        B_y2 -= alpha*(B_y2-B_y2_calc)

        B_y3_calc = (x[2]-x[3])/A_y3/R_y3
        B_y3 -= alpha*(B_y3-B_y3_calc)
        
        # Check for convergence
        B = np.array([B_r, B_y1, B_y2, B_y3])
        if np.linalg.norm(B-B_old)/np.linalg.norm(B) < tol:
            break
        B_old = B

    # Calculate flux
    phi_q1 = x[-2]/R_g1
    phi_q2 = x[4]/R_g2

    phi_q = phi_q1+phi_q2

    # Flux linkage
    lambda_q = constants.N_c*constants.k_w*phi_q

    # Inductance d-axis
    L_q = lambda_q/I_q

    # Logging
    if verbose == 3:
        print("--------------------- Q-axis Inductace ---------------------")
        print(f"B_r = {B_r}")
        print(f"B_y1 = {B_y1}")
        print(f"B_y2 = {B_y2}")
        print(f"B_y3 = {B_y3}")
        print(f"B_g2 = {x[4]/R_g2/A_g2}")
        print(f"B_g1 = {x[-2]/R_g1/A_g1}")
        print("")
        print(f"phi_r = {(x[1]-x[-3])/R_r}")
        print(f"phi_y1 = {(x[-3]-x[-2])/R_y1}")
        print(f"phi_y2 = {(x[1]-x[2])}")
        print(f"phi_y3 = {(x[2]-x[3])/R_y3}")
        print(f"phi_g2 = {x[4]/R_g2}")
        print(f"phi_g1 = {x[-2]/R_g1}")
        print("")
        print(f"R_r = {R_r}")
        print(f"R_y1 = {R_y1}")
        print(f"R_y2 = {R_y2}")
        print(f"R_y3 = {R_y3}")
        print(f"R_g2 = {R_g2}")
        print(f"R_g1 = {R_g1}")
        print("")
        print(f"lambda_q (total) = {lambda_q*constants.num_poles}")

    return L_q

# Maximum Torque Per Ampere (MTPA) angle
def opt_beta(lambda_pm, L_d, L_q, I_s):
    num1 = -lambda_pm+np.sqrt(lambda_pm**2+8*(L_d-L_q)**2*I_s**2)
    num2 = -lambda_pm-np.sqrt(lambda_pm**2+8*(L_d-L_q)**2*I_s**2)
    den = 4*(L_d-L_q)*I_s

    # Choose the positive beta that will produce a positive I_q
    beta1 = num1/den
    beta2 = num2/den

    return beta1 if beta1 > 0 else beta2

def torque_opt(p, I_s, iter = 10):
    # Initial guess
    beta = 30/180*np.pi

    # ------------------------------------------------
    old_beta = 0
    I_d = 0
    I_q = 0
    lambda_pm = 0
    for i in range(iter):
        I_d = -I_s*np.cos(beta)
        I_q = I_s*np.sin(beta)

        L_d= calculate_L_d(I_d, iter)*constants.num_poles
        L_q= calculate_L_q(I_q, iter)*constants.num_poles
        lambda_pm = solve_pm_equations()*constants.N_c*constants.num_poles

        beta = opt_beta(lambda_pm, L_d, L_q, I_s)

        # Stop if beta has converged
        if np.abs(beta-old_beta)/old_beta < 0.01:
            break

        if i%10 == 0 and verbose >= 2:
            print(f"Beta = {beta}")

        old_beta = beta
    # -----------------------------------------------
    if verbose >= 1:
        print(f"Final Beta: {beta:.4f}, I_d = {I_d:.2f}, I_q = {I_q:.2f}")
        print(f"Lambda_pm = {lambda_pm}, L_d = {L_d}, L_q = {L_q}")
    T = 3/2*p*(lambda_pm*I_q+(L_d-L_q)*I_d*I_q)

    return T

def torque_d_q(p, I_d, I_q, iter = 200):
    L_d = calculate_L_d(I_d, iter)*p*2
    L_q = calculate_L_q(I_q, iter)*p*2
    lambda_pm = solve_pm_equations()*constants.N_c*p*2
    if verbose >= 1:
        print("------------------------------------------")
        print(f"Lambda_pm = {lambda_pm}, L_d = {L_d}, L_q = {L_q}, F_q = {I_q*constants.N_c}, F_d = {I_d*constants.N_c}, I_d = {I_d}, I_q = {I_q}")
    T = 3/2*p*(lambda_pm*I_q+(L_d-L_q)*I_d*I_q)

    return T

def objective_func_theta(x):
    # Unpack x
    theta_pm = x[0]*theta_pm_max

    # Update motor
    constants.theta_pm = theta_pm
    constants.w_air_scaling = 0

    constants.magnet_cords = calculate_magnet_points()

    # Calculate torque
    beta = 0.7
    I_d = -constants.I_s*np.cos(beta)
    I_q = constants.I_s*np.sin(beta)
    T = float(torque_d_q(constants.p, I_d, I_q, 20))

    # Minimize the negative of the torque (maximize torque)
    v1 = magnet_length_constraint_func(-1)
    v2 = theta_pm_constraint_func(-1)

    if v1 < 0 and v2 < 0:
        return (v1**2+v2**2-T)
    elif v1 < 0:
        return (v1**2-T)
    elif v2 < 0:
        return (v2**2-T)
    return -T

def optimize_theta_pm():
    # Bounds
    bounds_theta_pm = [[10*np.pi/180/theta_pm_max, 1.0]]

    options = {
        "disp": True,
    }
 
    methods = ["Nelder-Mead", "trust-constr"]
    iterations = 5
    x0s = np.random.uniform(bounds_theta_pm[0][0], bounds_theta_pm[0][1], iterations) # Reuse initial conditions for both optimizations
    fig, axs = plt.subplots(2, 2)
    for j in range(len(methods)):
        for i in range(iterations):
            opt_vals = []
            xs = []

            def callback(intermediate_result: OptimizeResult):
                print(f"Intermediate Result: {intermediate_result}")
                try:
                    opt_vals.append(intermediate_result.fun)
                    xs.append(intermediate_result.x)
                except AttributeError:
                    print("Intermediate result did not have the fun attribute")


            # New initial condition each iteration
            x0 = [
                x0s[i]
            ]

            # Initial point is not marked in the callback function
            opt_vals.append(objective_func_theta(x0))
            xs.append(x0)
            
            res = minimize(
                objective_func_theta, 
                x0, 
                bounds=bounds_theta_pm, 
                options=options,
                method=methods[j],
                callback=callback,
            )
            
            axs[j, 0].plot(opt_vals, label=f"Initial Condition = {x0[0]*theta_pm_max*180.0/np.pi:.2f}°")
            axs[j, 1].plot(np.array(xs)*theta_pm_max*180.0/np.pi, label=f"Initial Condition = {x0[0]*theta_pm_max*180.0/np.pi:.2f}°")

        axs[j, 0].legend()
        axs[j, 1].legend()
        
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Best f(x)")
    axs[0, 0].set_ylabel("Best f(x)")
    axs[0, 0].set_title("Maximization of Torque w.r.t theta_pm (Nelder-Mead)")
    axs[1, 0].set_title("Maximization of Torque w.r.t theta_pm (trust-constr)")

    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Optimal theta_pm [°]")
    axs[0, 1].set_ylabel("Optimal theta_pm [°]")
    axs[0, 1].set_title("Optimal theta_pm over iterations (Nelder-Mead)")
    axs[1, 1].set_title("Optimal theta_pm over iterations (trust-constr)")

    for ax in axs.flat:
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_xmargin(0)

    plt.tight_layout()
    plt.show()


def objective_func(x):
    # Unpack x
    theta_pm = x[0]*theta_pm_max
    w_air_scaling = x[1]*w_air_max

    # Update motor
    constants.theta_pm = theta_pm
    constants.w_air_scaling = w_air_scaling

    constants.magnet_cords = calculate_magnet_points()

    # Calculate torque
    beta = 0.7
    I_d = -constants.I_s*np.cos(beta)
    I_q = constants.I_s*np.sin(beta)
    T = float(torque_d_q(constants.p, I_d, I_q, 20))

    # Minimize the negative of the torque (maximize torque)
    v1 = magnet_length_constraint_func(-1)
    v2 = theta_pm_constraint_func(-1)


    if v1 < 0 and v2 < 0:
        return (v1**2+v2**2-T)
    elif v1 < 0:
        return (v1**2-T)
    elif v2 < 0:
        return (v2**2-T)
    return -T

def greedy_opts():
    # Bounds
    bounds_theta_pm = [10*np.pi/180/theta_pm_max, 1.0]
    bounds_w_air = [0., 1.0]
    bounds = [bounds_theta_pm, bounds_w_air]

    options = {
        "disp": True,
    }
 
    # L-BFGS-B does not converge
    methods = ["Powell", "Nelder-Mead", "trust-constr"]

    # Intermediate results for plotting the history
    opt_vals = []
    def callback(intermediate_result: OptimizeResult):
        try:
            opt_vals.append(intermediate_result.fun)
        except AttributeError:
            print("Intermediate result did not have the fun attribute")

    # Final results
    converge_times = [[], [], []]
    opt_funcs = [[], [], []]
    opt_x = [[], [], []]

    # Perform optimization
    iterations = 10
    if iterations != 1:
        to_plot = False # Plot the history. True for plot. Meant for iterations = 1
    else:
        to_plot = True
    for _ in range(iterations):
        # New initial condition each iteration
        x0 = [
            np.random.uniform(bounds_theta_pm[0], bounds_theta_pm[1]),
            np.random.uniform(bounds_w_air[0], bounds_w_air[1]),
        ]

        for i in range(len(methods)):
            # Initial point is not marked in the callback function
            opt_vals.append(objective_func(x0))

            # Start optimization
            start_time = time.perf_counter()
           
            try:
                res = minimize(
                    objective_func, 
                    x0, 
                    bounds=bounds, 
                    options=options,
                    method=methods[i],
                    callback=callback,
                    tol=1e-5
                )
            
                end_time = time.perf_counter()
                total_duration = end_time - start_time

                # Log
                converge_times[i].append(total_duration)
                opt_funcs[i].append(res.fun)
                opt_x[i].append(res.x)
            except:
                print(f"ValueError: Initial condition = {x0}, method = {methods[i]}")
                converge_times[i].append(converge_times[i]/_)
                opt_funcs[i].append(opt_funcs[i]/_)
                opt_x[i].append(None)


            print(f"Optimization finished in {total_duration:.4f} seconds for method {methods[i]}, Optimal inputs: {res["x"]}")
            if to_plot:
                plt.plot(opt_vals, label=methods[i])
            opt_vals = []

    if to_plot:
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel("Iteration")
        plt.ylabel("Best f(x)")
        plt.legend()
        plt.margins(x=0)
        plt.show()

    for i in range(3):
        conv = np.array(converge_times[i])
        opt = np.array(opt_funcs[i])

        print(f"Mean convergence time: {np.mean(conv)}, std = {np.std(conv)}, method = {methods[i]}")
        print(f"Mean optimal value: {np.mean(opt)}, std = {np.std(opt)}, method = {methods[i]}")
        print(f"Optimal inputs: {opt_x[i]}")
        print(f"Optimal values: {opt}")
        print()

if __name__ == "__main__":
    beta = 0.7
    I_d = -constants.I_s*np.cos(beta)
    I_q = constants.I_s*np.sin(beta)
    constants.magnet_cords = calculate_magnet_points()
    print(torque_d_q(constants.p, I_d, I_q))
    """delta_time = []
    for i in range(100):
        start_time = time.perf_counter()
        beta = 0.7
        I_d = -constants.I_s*np.cos(beta)
        I_q = constants.I_s*np.sin(beta)
        constants.magnet_cords = calculate_magnet_points()
        print(torque_d_q(constants.p, I_d, I_q))
        end_time = time.perf_counter()
        delta_time.append(end_time - start_time)
    delta_time = np.array(delta_time)
    print(delta_time)
    print(np.mean(delta_time))
    print(np.std(delta_time))"""