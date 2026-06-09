import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import LHS
from pykrige import OrdinaryKriging
import json

plt.style.use("ggplot")

# LHS library: https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
# LHS requires the limits of the points to sample. In this tutorial, we'll only sample theta_pm which has limits [10 degrees; 75 degrees]


# Load data
with open("post_processed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process data
avg_torques = []
theta_pms = []
ripple_torques = []

for i in data:
    theta_pms.append(i["theta_pm"])
    avg_torques.append(i["avg_T"])
    ripple_torques.append(i["ripple_T"])

theta_pms = np.array(theta_pms)
avg_torques = np.array(avg_torques)
ripple_torques = np.array(ripple_torques)

# Kriging model
# The variogram model used is an exponential as it yields nonzero variance. This is in contrast to the Gaussian model, which can yield a "perfect" fit
uk_ripple_T = OrdinaryKriging(theta_pms, np.zeros(theta_pms.shape), ripple_torques, variogram_model="exponential", verbose=True)
uk_avg_T = OrdinaryKriging(theta_pms, np.zeros(theta_pms.shape), avg_torques, variogram_model="exponential", verbose=True)

# Predict
theta_pms_pred = np.linspace(np.min(theta_pms), np.max(theta_pms), 200)

y_pred, y_std = uk_avg_T.execute("grid", theta_pms_pred, np.array([0.0]))

y_pred = np.squeeze(y_pred)
y_std = np.squeeze(y_std)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.scatter(theta_pms, avg_torques, s=40, label="Input data")

ax.plot(theta_pms_pred, y_pred, label="Predicted values")
ax.fill_between(
    theta_pms_pred,
    y_pred - 3 * y_std,
    y_pred + 3 * y_std,
    alpha=0.3,
    label="Confidence interval",
)
ax.legend(loc=9)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()