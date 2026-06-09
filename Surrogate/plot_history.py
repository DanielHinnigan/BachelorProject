import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Imported ticker for integer formatting

plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or 'Computer Modern'
    "font.size": 15          # Match your thesis font size
})

if __name__ == "__main__":
    with open("bo_history_v5.json", "r") as f:
        data = json.load(f)
        
    EIs = []
    f_opts = []
    ripples = []
    avgs = []
    likelihoods = []
    fs = []
    variances = []
    means = []

    for row in data:
        EIs.append(row["Maximum EI"])
        f_opts.append(row["Current maximum"])
        likelihoods.append(row["Log-likelihood"])
        avgs.append(row["Average Torque"])
        ripples.append(row["Torque Ripple"])
        fs.append(row["Output"])
        variances.append(row["Predictive Variance"])
        means.append(row["Predictive mean"])
    
    # Convert from list to arrays
    avgs = np.array(avgs)
    np.array(fs)

    # Print parameters of best round
    idx = np.argmax(fs)
    print(data[idx])

    # Plot
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True, dpi=100)

    fontsize = 15
    # Plot: First row
    axs[0,0].plot(f_opts, "bo-")
    axs[0,0].set_ylabel("Max. Objective Function", fontsize=fontsize)

    axs[0,1].plot(EIs, "bo-")
    axs[0,1].set_ylabel("Expected Improvement", fontsize=fontsize)

    # Plot: Second row
    axs[1,0].plot(likelihoods, "bo-")
    axs[1,0].set_ylabel("Log-likelihood", fontsize=fontsize)

    axs[1,1].plot(-avgs, "bo-")
    axs[1,1].set_ylabel("Average Torque [Nm]", fontsize=fontsize)

    # Plot: Third row
    axs[2,0].plot(means, "bo-")
    axs[2,0].set_ylabel("Posterior Mean", fontsize=fontsize)

    axs[2,1].plot(variances, "bo-")
    axs[2,1].set_ylabel("Posterior Variance", fontsize=fontsize)

    # Plot: Fourth row
    axs[3,0].plot(ripples, "bo-")
    axs[3,0].set_ylabel("Torque Ripple [%]", fontsize=fontsize)
    axs[3,0].set_xlabel("Iteration", fontsize=fontsize)

    axs[3,1].plot(fs, "bo-")
    axs[3,1].set_ylabel("Objective Function", fontsize=fontsize)
    axs[3,1].set_xlabel("Iteration", fontsize=fontsize)

    # Force x-axis ticks to be integers only
    for ax in axs.flat:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show()