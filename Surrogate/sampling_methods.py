import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS, Random, FullFactorial

# Configuration
xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])
num_points = 25

# Random
rand_gen = Random(xlimits=xlimits)
x_random = rand_gen(num_points)

# LHS (using 'center' to match your previous question)
lhs_gen = LHS(xlimits=xlimits, criterion='center')
x_lhs = lhs_gen(num_points)

# Full Factorial (requires a perfect square for 2D, e.g., 25 points)
ff_gen = FullFactorial(xlimits=xlimits)
x_ff = ff_gen(num_points)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

methods = [
    (x_random, 'Random Sampling', 'firebrick'),
    (x_lhs, 'Latin Hypercube Sampling (LHS)', 'royalblue'),
    (x_ff, "Grid Sampling", 'forestgreen')
]

for i, (data, title, color) in enumerate(methods):
    axes[i].scatter(data[:, 0], data[:, 1], color=color, edgecolors='k', zorder=3)
    axes[i].set_title(f"{title}\n({len(data)} points)", fontsize=14, fontweight='bold')
    
    # Add grid lines to show the "bins" for LHS visualization
    grid_lines = np.linspace(0, 1, num_points + 1)
    if 'LHS' in title:
        for line in grid_lines:
            axes[i].axvline(line, color='gray', lw=0.5, alpha=0.3)
            axes[i].axhline(line, color='gray', lw=0.5, alpha=0.3)
            
    axes[i].set_xlim(-0.05, 1.05)
    axes[i].set_ylim(-0.05, 1.05)
    axes[i].set_aspect('equal')
    axes[i].set_xlabel('Variable X1')
    if i == 0: axes[i].set_ylabel('Variable X2')

plt.tight_layout()
plt.savefig('sampling_comparison.png')
plt.show()