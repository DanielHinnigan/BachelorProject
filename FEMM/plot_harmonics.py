import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or 'Computer Modern'
    "font.size": 11          # Match your thesis font size
})

def harmonic(h):
    # Parameters
    gamma = 30*np.pi/180
    Q = 2

    if h%2 == 1:
        return np.sin(h*Q*gamma/2)/(Q*np.sin(h*gamma/2))
    else:
        return 0
    
def mmf_mag(k, h):
    return k/h

if __name__ == "__main__":
    ns = np.arange(1, 31)
    harmonics = np.array([harmonic(n) for n in ns])
    print(harmonics)
    fig, axs = plt.subplots(1, 2)

    axs[0].stem(ns, harmonics)
    axs[0].set_xlabel("Harmonic Order ($h$)")
    axs[0].set_ylabel("Magnitude of winding factor")
    axs[0].set_title("First 30 winding factors of the winding configuration")

    axs[1].stem(ns, mmf_mag(harmonics, ns), label="Toyota Winding Configuration")
    markerline, stemlines, _ = axs[1].stem(ns, mmf_mag(np.tile([1, 0], 15), ns), linefmt="b-", label="Full-Pitch, Concentrated Winding", basefmt=" ")
    plt.setp(stemlines, alpha=0.3, color='blue', linewidth=1)
    plt.setp(markerline, alpha=0.6, color='blue', marker='x')
    axs[1].set_xlabel("Harmonic Order ($h$)")
    axs[1].set_ylabel("Magnitude of mmf harmonic")
    axs[1].set_title("First 30 harmonics of the winding configuration")
    axs[1].legend()

    plt.tight_layout()
    plt.show()