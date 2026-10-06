import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Ensure figs directory exists
os.makedirs("figs", exist_ok=True)

# Use LaTeX font with a moderate size
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 24  # Adjust as needed
})

def plot_pbmc_forecasting_maintext(Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                   outname="pbmc_forecasting_maintext.pdf", cmap_str="coolwarm", size_points=10):
    """
    Main text figure: 1 row x 8 columns.
      - First 4: Training points 1, 7, 14, 20 (indices 0, 6, 13, 19)
      - Next 4: [Ground Truth | Ours | SBIRR-ref | SB-forward] for the forecast step
    Axes are fixed across all subplots for comparability.
    """
    train_indices = [0, 6, 13, 19]
    nrows, ncols = 1, 8

    # Compute axis limits across all data
    all_data = np.concatenate([Xs[i] for i in train_indices] + [X_val, forecast[1], forecastsbirr[1], forecastsbforward[1]], axis=0)
    xlim = (-.3, 2)
    ylim = (-.5, .75)
    zlim = (0.5, 2)
    cmap_slices = cm.get_cmap(cmap_str)

    fig = plt.figure(figsize=(32, 4))  # Wide and compact
    axs = []
    for j in range(ncols):
        ax = fig.add_subplot(1, ncols, j+1, projection='3d')
        axs.append(ax)

    # First 4: selected training points
    for j, idx in enumerate(train_indices):
        ax = axs[j]
        color_i = cmap_slices(idx / 21)
        ax.scatter(Xs[idx][:, 0], Xs[idx][:, 1], Xs[idx][:, 2], color=color_i, marker="x", s=size_points*2)
        ax.set_title(f"Step {idx+1}", fontsize=54)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("", labelpad=0, fontsize=20)
        ax.set_ylabel("", labelpad=0, fontsize=20)
        ax.set_zlabel("", labelpad=0, fontsize=20)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(-140, -140)

    # Next 4: forecasts
    forecast_methods = [
        ("Truth", X_val),
        ("Ours", forecast[1]),
        ("SBIRR-ref", forecastsbirr[1]),
        ("SB-forward", forecastsbforward[1])
    ]
    for j, (label, data) in enumerate(forecast_methods):
        ax = axs[4+j]
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=cmap_slices(0.95), marker="o", s=size_points*2)
        ax.set_title(label if label == "Truth" else f"{label}", fontsize=54)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("", labelpad=0, fontsize=20)
        ax.set_ylabel("", labelpad=0, fontsize=20)
        ax.set_zlabel("", labelpad=0, fontsize=20)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(-140, -140)

    plt.tight_layout()
    # Add a vertical line between the 4th and 5th subplot
    fig.canvas.draw()  # Needed to ensure positions are correct
    # Get bounding boxes for the axes
    bbox1 = axs[3].get_position()
    bbox2 = axs[4].get_position()
    # x position between subplots 4 and 5
    x_split = (bbox1.x1 + bbox2.x0) / 2
    line = plt.Line2D([x_split, x_split], [0.05, 0.95], color='black', linewidth=3, transform=fig.transFigure, zorder=100)
    fig.add_artist(line)
    plt.savefig(f"./figs/{outname}", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from make_plot import get_data

    # You can adjust these parameters as needed
    task_name = "pbmc"
    seed = 42
    kind = "realdata"
    output_dir = os.path.join("figs", "pbmc_progression")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
    model, sbirr, sbforward, time_scale, vector_fields) = get_data("pbmc", seed, "realdata")
    # Save figure for main text
    plot_pbmc_forecasting_maintext(
        Xs, X_val, forecast, forecastsbirr, forecastsbforward,
        outname=os.path.join("pbmc_progression", "pbmc_forecasting_maintext.pdf")
    )
