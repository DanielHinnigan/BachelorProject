import numpy as np
import matplotlib.pyplot as plt

# Create a 2D NumPy array
data_femm = np.abs(np.loadtxt("data/datapoints_femm_4poles_theta_pms.txt"))

data_mec = np.loadtxt("data/datapoints_MECv5_4poles_theta_pms.txt")

normalized = True

if normalized:
    plt.scatter(data_femm[:,0], data_femm[:,1]/np.max(data_femm[:,1]), color="blue")
    plt.plot(data_femm[:,0], data_femm[:,1]/np.max(data_femm[:,1]), color="blue", label="FEMM")

    plt.scatter(data_mec[:,0]*180/np.pi, data_mec[:,1]/np.max(data_mec[:,1]), color="orange")
    plt.plot(data_mec[:,0]*180/np.pi, data_mec[:,1]/np.max(data_mec[:,1]), color="orange", label="MEC")

    plt.grid()
    plt.xlabel("Angle between magnets (degrees)")
    plt.ylabel("Normalized Torque")
    plt.title("Normalized Torque as a function of the angle between magnets for 12 slots, 4 poles PMSM")
else:
    plt.scatter(data_femm[:,0], data_femm[:,1], color="blue")
    plt.plot(data_femm[:,0], data_femm[:,1], color="blue", label="FEMM")

    plt.scatter(data_mec[:,0]*180/np.pi, data_mec[:,1], color="orange")
    plt.plot(data_mec[:,0]*180/np.pi, data_mec[:,1], color="orange", label="MEC")

    plt.grid()
    plt.xlabel("Angle between magnets (degrees)")
    plt.ylabel("Torque (Nm)")
    plt.title("Torque as a function of the angle between magnets for 12 slots, 4 poles PMSM")

plt.legend()
plt.show()