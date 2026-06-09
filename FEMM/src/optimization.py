import numpy as np
from src.constants import *
from scipy.optimize import minimize
from src.constraint_funcs import *
import pygad

# ------------------------------------------------------------------------------------------------

def torque(currents, d_lambdas):
    return np.sum([currents[i]*d_lambdas for i in range(3)])

# ------------------------------------------------------------------------------------------------

def d_theta_PHI(theta, theta_r, A_m, B_r, pole_pairs):
    theta_shifted = (theta-theta_r)%(2*np.pi)

    d_PHI = np.where(((theta_shifted*2*pole_pairs/(2*np.pi))%2) < 1, 2*A_m*B_r, -2*A_m*B_r)

    return d_PHI        

def turns_function(steps, amplitude):
    # steps is an array with element 0 = no step, element 1 = step up, and element -1 = step down
        # Adding more zeros decreases the width of steps[i]-steps[i-1] w.r.t theta
    # The amplitude is dependent on the phase
    # Step function is assummed equal for each step (MAYBE NOT POSSIBLE!!!!!!!!!!!!!!!!!!!!!!!!!!)

    # This function does not enforce that the turns function is periodic with 2pi, however,
    # this is expected to be enforced by the optimization algorithm

    # Three phases
    turns_funcs = np.array([np.cumsum(steps[i]*amplitude[i]) for i in range(3)])

    return turns_funcs

def d_theta_lambdas(theta_r, A_m, B_r, pole_pairs, currents, time, steps):
    theta = np.linspace(0, 2*np.pi, N_theta)
    
    # Amplitudes are based on currents and number of turns
    amplitudes = currents*n_turns*np.cos(omega_e*time-phase)

    Ns = turns_function(steps, amplitudes)
    d_PHI = d_theta_PHI(theta, theta_r, A_m, B_r, pole_pairs)
    d_lambdas = np.array([Ns[j]*d_PHI for j in range(3)])

    return d_lambdas

# ------------------------------------------------------------------------------------------------

def objective_function(params_dict):
    """
    params_dict has the same form as the dict "all_paramters" (look in constants.py)
    """
    
    # Cross-sectional area of magnet in m^2
    # Note: Any sizes are given in mm by default
    A_m = length*params_dict["magnet length"]*10**(-6) 

    currents = np.array([params_dict["i_A"], params_dict["i_B"], params_dict["i_C"]])
    steps = [params_dict["turns_step_A"], params_dict["turns_step_B"], params_dict["turns_step_C"]]

    # First term: Maximize torque production
    d_lambdas=d_theta_lambdas(params_dict["theta_r"], A_m, 
                              params_dict["magnet strength"],
                              params_dict["pole pairs"],
                              currents,
                              params_dict["time"],
                              steps
                              )
    T = torque(currents, d_lambdas)

    # Second term: Force induced emf to be sinusoidal
    term2 = np.sum([(np.cos(omega_e*params_dict["time"]-phase[j])-d_lambdas[j]/np.max(d_lambdas[j]))**2 for j in range(3)])

    # Constraints:
    penalization = 0
    if (theta_pm_constraint_func(None) < 0):
        penalization += 1e10/theta_pm_constraint_func(None)
    if (magnet_length_constraint_func(None) < 0):
        penalization += 1e10/magnet_length_constraint_func(None)
    if (np.sum(params_dict["turns_step_A"]) != 0):
        # Sum of currents in and out of the plane must sum to 0
        penalization += 1e9*np.abs(np.sum(params_dict["turns_step_A"]))
    if (np.sum(params_dict["turns_step_B"]) != 0):
        # Sum of currents in and out of the plane must sum to 0
        penalization += 1e9*np.abs(np.sum(params_dict["turns_step_B"]))
    if (np.sum(params_dict["turns_step_C"]) != 0):
        # Sum of currents in and out of the plane must sum to 0
        penalization += 1e9*np.abs(np.sum(params_dict["turns_step_C"]))

    # Collect terms
    fitness = k_T*T-k_f*term2-penalization
    return fitness

# -----------------------------------------------------------

# WINDING FUNCTION CONSTRAINT OR AT LEAST IMPLEMENTATION!!!!!
# Main function
def optimize(all_parameters, to_tune):
    # Convert key-value pairs to values such that minimize() can handle the initial parameters
    x0 = np.array([all_parameters[parameter] for parameter in to_tune])

    # Wrapper function to make the objective function compliant with minimize()
    def wrapper(x):
        # Update parameters that are set to be optimized for
        current_params = all_parameters.copy()
        for i, key in enumerate(to_tune):
            current_params[key] = x[i]

        # Number of pole pairs must be integer valued
        current_params["pole pairs"] = np.round(current_params["pole pairs"])

        # Use the actual objective function to optimize
        return objective_function(current_params)

    print("Starting minimize")

    # Minize
    minimize(wrapper, x0, bounds = bounds, constraints = constraints, method="COBYLA", 
             options = {"disp": True, "maxiter": 10})

def optimize_GA():

    # Wrapper function to make the objective function compliant with ga_instace()
    def wrapper(ga_instance, solution, solution_idx):
        # Logging
        print(f"Solution index: {solution_idx}")

        # Update parameters that are set to be optimized for
        current_params = all_parameters.copy()
        for i, key in enumerate(to_tune_ga):
            if key[:-2] == "turns_step":
                turns = []
                for j in range(num_bins_turns):
                    turns.append(solution[i+j])
                current_params[key] = np.array(turns)
            else:
                current_params[key] = solution[i]

        # Use the actual objective function to optimize
        fitness = objective_function(current_params)
        if solution_idx%5 == 0:
            print(f"Fitness: {fitness}\tSolution: \n{solution}")
        return fitness

    print("Instantiating GA")
    ga_instance = pygad.GA(
        num_generations=1000,
        num_parents_mating=10,
        fitness_func=wrapper,
        sol_per_pop=101,
        num_genes=total_genes,
        gene_space=gene_space,
        gene_type = [float, float, float, float, float, float]+[int]*3*num_bins_turns+[int],
        mutation_percent_genes=10
    )

    print("Running GA")
    ga_instance.run()