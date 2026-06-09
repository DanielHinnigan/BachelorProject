import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

from src.constants import g, inner_R_stator

plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or 'Computer Modern'
    "font.size": 11          # Match your thesis font size
})

def load_data(file_path):
    try:
        # loadtxt ignores lines starting with '%' by default
        # It reads the scientific notation (e.g., 9.17490954e+000) automatically
        data = np.loadtxt(file_path)

        # Slice the data into two separate arrays
        # Column 0: Length (mm)
        length_array = data[:, 0]
        
        # Column 1: H.n (Amp/m)
        hn_array = data[:, 1]

        # Verification: Print the first 5 entries of each
        print(f"Successfully imported {len(length_array)} data points.")
        print("\nFirst 5 entries of Length (mm):")
        print(length_array[:5])
        
        print("\nFirst 5 entries of H.n (Amp/m):")
        print(hn_array[:5])

        return length_array, hn_array
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def theoretical(N, angles: np.ndarray):
    n_turns = 8 # Turns per coil
    n_coils = 2 # coils per pole
    n_p = 4 
    N_t = n_turns*n_coils*n_p # Fits very well, needs an argument
    Q = 2
    gamma = 30*np.pi/180

    mmf = np.zeros(angles.shape)
    for h in range(1,N+1,2):
        k_dh = np.sin(h*Q*gamma/2)/(Q*np.sin(h*gamma/2))
        mmf += 1/h*np.sin(h*angles*4)*k_dh
    #mmf *= 4/np.pi*N_t*100/8
    mmf *= 4/np.pi*16/2*100 # 16 turns per pole (two coils with 8 turns each). Divide by 2 due to the derivation (pretty easy, follow https://www.youtube.com/watch?v=5ND1qhV1OU0&t=77s)

    return mmf



if __name__ == "__main__":
    # Normalize harmonics
    normalize = True

    # Import data
    l_stator, bn_stator = load_data("H_fields/B_stator_ideal.txt")
    l_rotor, hn_rotor = load_data("H_fields/H_rotor_ideal.txt")

    l_stator_practice, hn_stator_practice = load_data("H_fields/H_stator_actual.txt")
    l_rotor_practice, hn_rotor_practice = load_data("H_fields/H_rotor_actual.txt")


    # Convert data to useful data
    hn_stator = bn_stator/(4*np.pi*10**(-7))

    mmf_femm = g*(hn_stator-hn_rotor)*10**(-3)
    mmf_femm -= np.mean(mmf_femm)
    angles = l_stator/(2*np.pi*inner_R_stator)*2*np.pi


    mmf_femm_practice = g*(hn_stator_practice-hn_rotor_practice)*10**(-3)
    mmf_femm_practice -= np.mean(mmf_femm_practice)
    angles_practice = l_stator_practice/(2*np.pi*inner_R_stator)*2*np.pi

    # Experimental harmonics
    inc_angle = angles[1]-angles[0]
    N = len(mmf_femm)

    harmonics_femm = fft(mmf_femm)
    freqs = fftfreq(N, inc_angle)[:N//2]/4*2*np.pi # Only take positive frequencies
    amp_femm = np.abs(2.0/N*harmonics_femm[0:N//2])

    harmonics_femm_practice = fft(mmf_femm_practice)
    N_practice = len(mmf_femm_practice)
    freqs_practice = fftfreq(N_practice, angles_practice[1]-angles_practice[0])[:N_practice//2]/4*2*np.pi
    amp_femm_practice = np.abs(2.0/N_practice*harmonics_femm_practice[0:N_practice//2])

    # Theoretical
    mmf_theory = theoretical(1024, angles)    
    harmonics_theory = fft(mmf_theory)
    amp_theory = np.abs(2.0/N*harmonics_theory[0:N//2])

    # Create a mask for significant values (ignore the 'noise' at 0)
    threshold = 0.01 * np.max(amp_theory) 
    mask = (amp_theory > threshold) | (amp_femm > threshold)

    mask_practice = amp_femm_practice > (0.01)*np.max(amp_femm_practice)

    # Plot
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(angles, mmf_femm, label="FEMM MMF")
    axs[0].plot(angles, mmf_theory, label="Theoretical MMF")
    axs[0].set_xlabel("Mechanical Angles [rad]")
    axs[0].set_ylabel("Airgap MMF [A]")
    axs[0].set_title("Comparison of MMF from theory and FEMM")
    axs[0].legend()

    if normalize:
        axs[1].stem(freqs[mask], amp_theory[mask]/np.max(amp_theory), linefmt="r-", markerfmt="ro", label="Theoretical")
        axs[1].stem(freqs[mask], amp_femm[mask]/np.max(amp_femm), linefmt='b-', markerfmt='bx', label='FEMM Ideal', basefmt=" ")
        axs[1].stem(freqs_practice[mask_practice], amp_femm_practice[mask_practice]/np.max(amp_femm_practice), linefmt='g-', markerfmt='gx', label='FEMM Practice', basefmt=" ")
        axs[1].set_ylabel("Normalized Amplitude of MMF Harmonics")
    else:
        axs[1].stem(freqs[mask], amp_theory[mask], linefmt="r-", markerfmt="ro", label="Theoretical")
        axs[1].stem(freqs[mask], amp_femm[mask], linefmt='b-', markerfmt='bx', label='FEMM Ideal', basefmt=" ")
        axs[1].stem(freqs_practice[mask_practice], amp_femm_practice[mask_practice], linefmt='g-', markerfmt='gx', label='FEMM Practice', basefmt=" ")
        axs[1].set_ylabel("Harmonics Amplitude")
    axs[1].set_xlim(0, 20)
    axs[1].set_xlabel("Harmonic Order")
    axs[1].legend()
    axs[1].set_xticks(np.arange(0,21,1))
    axs[1].set_title("MMF Harmonics comparison between FEMM and Theory")

    plt.show()