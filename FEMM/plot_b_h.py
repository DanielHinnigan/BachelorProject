import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline


plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "figure.figsize": (6, 4),    # Set dimensions in inches
    "font.size": 200,             # Matches standard paper font size
    "axes.labelsize": 16,
    "legend.fontsize": 20,
    "savefig.dpi": 300,          # High resolution
    #"text.usetex": True,         # Renders text with LaTeX (requires local TeX installation)
    "font.family": "serif"       # Matches Times New Roman or Computer Modern
})
plt.rcParams['lines.linewidth'] = 3.0

# Steel (M-19 Steel)
H_steel = np.array([0.000000, 15.120714, 22.718292, 27.842733, 31.871434, 35.365044, 38.600588, 41.736202, 44.873979, 48.087807, 51.437236, 54.975221, 58.752993, 62.823644, 67.245285, 72.084406, 77.420100, 83.350021, 89.999612, 97.537353, 106.201406, 116.348464, 128.547329, 143.765431, 163.754169, 191.868158, 234.833507, 306.509769, 435.255202, 674.911968, 1108.325569, 1813.085468, 2801.217421, 4053.653117, 5591.106890, 7448.318413, 9708.815670, 12486.931615, 16041.483644, 21249.420624, 31313.495878, 53589.446877, 88477.484601, 124329.410540, 159968.569300, 197751.604272, 234024.751347])
B_steel = np.array([0.000000, 0.050000, 0.100000, 0.150000, 0.200000, 0.250000, 0.300000, 0.350000, 0.400000, 0.450000, 0.500000, 0.550000, 0.600000, 0.650000, 0.700000, 0.750000, 0.800000, 0.850000, 0.900000, 0.950000, 1.000000, 1.050000, 1.100000, 1.150000, 1.200000, 1.250000, 1.300000, 1.350000, 1.400000, 1.450000, 1.500000, 1.550000, 1.600000, 1.650000, 1.700000, 1.750000, 1.800000, 1.850000, 1.900000, 1.950000, 2.000000, 2.050000, 2.100000, 2.150000, 2.200000, 2.250000, 2.300000])

# Magnets (42SH)
B_m = np.array([0.000000, 0.169884,0.322394,0.405405,0.474903,0.610039,0.805019,1.003860,1.094590])
H_m = np.array([0.000000,3645.000000,18224.000000,52849.000000,98409.000000,196818.000000,344431.000000,495689.000000,566762.000000])

# Interpolaters
interpolate_m = CubicSpline(B_m, H_m, bc_type="natural", extrapolate=True)
interpolate_steel = CubicSpline(B_steel, H_steel, bc_type="natural", extrapolate=True)

points_steel = np.linspace(B_steel[0], B_steel[-1], 1000)
points_m = np.linspace(B_m[0], B_m[-1], 1000)

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(interpolate_steel(points_steel), points_steel, color="black")
ax1.scatter(H_steel, B_steel, color="black")
ax1.set_ylabel("B [T]")
ax1.set_xlabel("H [A/m]")
ax1.grid(True)
ax1.set_title("M250-35A Steel", fontsize=20)
ax1.set_ylim(B_steel[0], B_steel[-1]*1.05)
ax1.set_xlim(None, H_steel[-1])

ax2.plot(interpolate_m(points_m), points_m, color="black")
ax2.scatter(H_m, B_m, color="black")
ax2.set_ylabel("B [T]")
ax2.set_xlabel("H [A/m]")
ax2.grid(True)
ax2.set_title("BNM-42SH  Magnet", fontsize=20)
ax2.set_ylim(B_m[0], B_m[-1]*1.05)
ax2.set_xlim(None, H_m[-1])

plt.show()