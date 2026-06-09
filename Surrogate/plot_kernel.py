import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or 'Computer Modern'
    "font.size": 15          # Match your thesis font size
})


def rbf_anisotropic(xi: np.ndarray, xj: np.ndarray, M: np.ndarray):
    return np.exp(-(xi-xj).T@M@(xi-xj))

def rbf_isotropic(xi, xj, l):
    return np.exp(-np.linalg.norm(xi-xj)**2/(2*l**2))

if __name__ == "__main__":
    # Grid
    xi = np.linspace(-1, 1, 100)
    xj = np.linspace(-1, 1, 100)

    XI, XJ = np.meshgrid(xi, xj)

    # Populate kernels
    K_iso = np.empty(XJ.shape)
    M = np.array([
        [1/2, 0],
        [0, 1/4]
    ])
    K_an = np.empty(XJ.shape)
    l1 = 1.
    point = np.array([0., 0.0])
    for i in range(len(xi)):
        for j in range(len(xj)):
            v = np.array([xi[i], xj[j]])
            K_iso[i,j] = rbf_isotropic(point, v, l1)
            K_an[i,j] = rbf_anisotropic(point, v, M)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained', sharey=True)

    global_min = np.min((np.min(K_an), np.min(K_iso)))
    global_max = np.max((np.max(K_an), np.max(K_iso)))

    cmap_shared = plt.cm.viridis

    # First plot
    im1 = axs[0].contourf(XI, XJ, K_iso, cmap=cmap_shared, vmin=global_min, vmax=global_max)
    axs[0].set_ylabel("Second Input Dimension")
    axs[0].set_xlabel("First Input Dimension")
    axs[0].set_title(fr"Isotropic Kernel with $\ell$ = {l1}")
    axs[0].scatter(*point)

    # Second plot
    im2 = axs[1].contourf(XI, XJ, K_an, cmap=cmap_shared, vmin=global_min, vmax=global_max)
    axs[1].set_xlabel("First Input Dimension")
    axs[1].set_title(f"Anisotropic RBF Kernel with $M_1$ = {M[0,0]} and $M_2$ = {M[1,1]}")
    axs[1].scatter(*point)

    # Fix aspect ratio
    for ax in axs:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=axs, label='Kernel Value', shrink=0.8, aspect=25, pad=0.02)

    # Final plot
    plt.show()