import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_ei(inc, sigma):
    """Calculate Expected Improvement for a grid of mu and sigma."""
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    z = inc/ sigma
    ei = np.maximum(inc, 0)+sigma*norm.pdf(z)-np.abs(inc)*norm.cdf(z)
    ei = inc*norm.cdf(inc/sigma)+sigma*norm.pdf(inc/sigma)
    return ei

# 1. Setup Grid
inc_range = np.linspace(-5., 5., 100)      # Predictive Mean
sigma_range = np.linspace(0.0, 10., 100)   # Predictive std

INC, SIGMA = np.meshgrid(inc_range, sigma_range)
EI = calculate_ei(INC, SIGMA)

# 2. Plotting
plt.figure(figsize=(10, 7))
levels = 11
contour = plt.contourf(SIGMA, INC, EI, levels=levels, cmap='viridis')
plt.colorbar(contour)

# Add lines to show specific EI thresholds
lines = plt.contour(SIGMA, INC, EI, levels=levels, colors='white', alpha=0.3)
plt.clabel(lines, inline=True, fontsize=8)

plt.title('Expected Improvement', fontweight='bold', fontsize=18)
plt.xlabel('$\sigma_n(x)$', fontsize=14)
plt.ylabel('$\mu_n(x)-f^\star$', fontsize=14)

plt.show()
plt.savefig("expected_improvement.jpg")