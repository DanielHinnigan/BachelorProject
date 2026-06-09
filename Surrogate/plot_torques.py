import matplotlib.pyplot as plt
import numpy as np
import json

plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or 'Computer Modern'
    "font.size": 15          # Match your thesis font size
})

if __name__ == "__main__":
    # Import data
    with open("data_LHS_12.json", "r") as f:
        random = json.load(f)[0]
    with open("data_ne.json", "r") as f:
        deterministic = json.load(f)[0]

    # Extract data
    torques_random = random["torques"]
    angles_random = random["rot_angles"]
    torques_det = deterministic["torques"]
    angles_det = deterministic["rot_angles"]

    # Plot
    plt.scatter(angles_random, torques_random, label="Random Sampling", color="b")
    plt.plot(angles_random, torques_random, linestyle="--", color="b")
    
    plt.scatter(angles_det, torques_det, label="Deterministic Sampling", color="orange")
    plt.plot(angles_det, torques_det, linestyle="--", color="orange")

    plt.title("Comparison of torque between random sampling and deterministic sampling")
    plt.ylabel("Torque [Nm]")
    plt.xlabel("Rotational Angle [°]")
    plt.legend()
    plt.show()