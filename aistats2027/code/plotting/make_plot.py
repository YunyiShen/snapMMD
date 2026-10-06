import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

###############################################################################
# 1) Create "plots" folder if it doesn't exist
###############################################################################
os.makedirs("figs", exist_ok=True)

###############################################################################
# 2) Use LaTeX font with a moderate size
###############################################################################
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 24  # Adjust as needed
})

###############################################################################
# Get data function 
###############################################################################
def get_data(task_name, seed=42, kind="classic"):
    data = np.load(f"../{kind}/data/{task_name}_data.npz")
    if "pbmc" in task_name:
        data = np.load(f"../{kind}/data/processed_pbmc_data_sub500_every_2_until20.npz")
    N_steps = data['N_steps']
    time_scale = data['time_scale']
    Xs = [data["Xs"][i] for i in range(N_steps - 1)]  # training data slices
    X_val = data["Xs"][-1]                            # final forecast slice

    forecast = np.load(f"../{kind}/forecasts/{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbirr = np.load(f"../{kind}/forecasts/SBIRR_{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbforward = np.load(f"../{kind}/forecasts/SBforward_{task_name}_forecast_{seed}.npz")['forecast']
    if "pbmc" in task_name:
        vector_fields = None
    else:
        vector_fields = np.load(f"../{kind}/vector_fields/{task_name}_vector_field_{seed}.npz")

    # Dummy placeholders
    model = 1
    sbirr = 1
    sbforward = 1

    return Xs, X_val, forecast, forecastsbirr, forecastsbforward, \
           model, sbirr, sbforward, time_scale, vector_fields

###############################################################################
# ===================== 2D: Forecasting Figure (no vector fields) =============
###############################################################################
def plot_forecasting_2d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname="LV_forecasting.png", cmap_str="coolwarm", size_points=30):
    """
    Single-row, 4-column figure:
      [Ground Truth | Ours | SBIRR-ref | SB-forward].
    No vector fields – just training scatter plus final forecast scatter.
    A colorbar (for training slice “time”) is placed in its own Axes on the right.
    A legend (x = training, o = forecast) is centered below.
    """
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    # Leave space for colorbar: subplots span x=0.06 to 0.86.
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.2, top=0.88, wspace=0.3)
    # Create an extra Axes for the colorbar on the far right.
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.68])  # spans same vertical region as subplots

    cmap_slices = cm.get_cmap(cmap_str)
    n_slices = len(Xs)
    for i, X in enumerate(Xs):
        color_i = cmap_slices(i / (n_slices + 1))
        for j in range(4):
            ax[j].scatter(X[:, 0], X[:, 1], color=color_i, marker="x", s=size_points)
    forecast_color = cmap_slices(0.95)
    ax[0].scatter(X_val[:, 0], X_val[:, 1], color=forecast_color, marker="o", s=size_points)
    ax[0].set_title("Ground Truth")
    ax[1].scatter(forecast[1, :, 0], forecast[1, :, 1], color=forecast_color, marker="o", s=size_points)
    ax[1].set_title("Ours")
    ax[2].scatter(forecastsbirr[1, :, 0], forecastsbirr[1, :, 1], color=forecast_color, marker="o", s=size_points)
    ax[2].set_title("SBIRR-ref")
    ax[3].scatter(forecastsbforward[1, :, 0], forecastsbforward[1, :, 1], color=forecast_color, marker="o", s=size_points)
    ax[3].set_title("SB-forward")

    if task_name == "LV":
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.set_ylim([0.75, 5])
            a.set_xlabel("Prey")
        ax[0].set_ylabel("Predator")
    if task_name == "GoM":
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.set_ylim([-1.5, 1.1])
            a.set_xlim([-1.5, 1.])
            a.set_xlabel("Longitude")
        ax[0].set_ylabel("Latitude")

    # Create colorbar for training slices.
    norm_train = mcolors.Normalize(vmin=0, vmax=n_slices)
    sm_train = cm.ScalarMappable(norm=norm_train, cmap=cmap_slices)
    sm_train.set_array([])
    cbar = fig.colorbar(sm_train, cax=cbar_ax)
    cbar.set_label("Time")

    # Legend: training marker (x) vs forecast marker (o)
    import matplotlib.lines as mlines
    train_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                 markersize=8, label='Training')
    forecast_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=8, label='Forecast')
    fig.legend(handles=[train_marker, forecast_marker],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)

    plt.savefig(f"./figs/{outname}", dpi=300)
    # plt.show()
    # plt.close(fig)


###############################################################################
# ===================== 2D: Appendix Figure (3 rows: forecast, VF, diff VF) ===
###############################################################################
def plot_appendix_2d_diff_arrows(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname="LV_appendix_diff_arrows.png", size_points=30):
    """
    3-row, 4-col 2D figure:
      Row 1: Forecasting scatter (with training data colored by “Time” using colormap A).
      Row 2: Actual vector fields (black arrows).
      Row 3: Difference vector fields = (v_method - v_GT), where arrows are colored
             according to their magnitude using colormap B.
    The same arrow scale is used in Rows 2 and 3.
    Two separate colorbars (one for Row 1 and one for Row 3) are added in custom Axes;
    here they are sized to be 66% of the row height and aligned with the rows.
    A legend for Row 1 (x = training, o = forecast) is added below.
    """
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    # Set subplots region: leaving space on right for two colorbars.
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.12, top=0.88, wspace=0.4, hspace=0.3)
    # For two colorbars, we add two Axes:
    # Assume subplots vertical span is from 0.12 to 0.88 (0.76 total). In a 3-row grid, each row ~0.253.
    # We want each colorbar to be 66% of a row height (~0.167) and centered vertically on its row.
    # Top row center ~ 0.88 - 0.1267 = 0.7533 → bottom = 0.7533 - 0.0833 = 0.67.
    # Bottom row center ~ 0.12 + 0.1267 = 0.2467 → bottom = 0.2467 - 0.0833 = 0.1634.
    cbar_ax_1 = fig.add_axes([0.88, 0.7, 0.02, 0.167])  # for Row 1
    cbar_ax_2 = fig.add_axes([0.88, 0.133, 0.02, 0.167]) # for Row 3

    row1 = axes[0, :]
    row2 = axes[1, :]
    row3 = axes[2, :]

    # Use two different colormaps:
    cmap_slices = cm.get_cmap("coolwarm")  # for Row 1 (training slices)
    cmap_diff = cm.get_cmap("plasma")       # for Row 3 (difference magnitude)

    n_slices = len(Xs)
    # Row 1: Forecast scatter
    for i, X in enumerate(Xs):
        color_i = cmap_slices(i / (n_slices + 1))
        for col in range(4):
            row1[col].scatter(X[:, 0], X[:, 1], color=color_i, marker="x", s=size_points)
    forecast_color = cmap_slices(0.95)
    row1[0].scatter(X_val[:, 0], X_val[:, 1], color=forecast_color, marker="o", s=size_points)
    row1[0].set_title("Ground Truth")
    row1[1].scatter(forecast[1, :, 0], forecast[1, :, 1], color=forecast_color, marker="o", s=size_points)
    row1[1].set_title("Ours")
    row1[2].scatter(forecastsbirr[1, :, 0], forecastsbirr[1, :, 1], color=forecast_color, marker="o", s=size_points)
    row1[2].set_title("SBIRR-ref")
    row1[3].scatter(forecastsbforward[1, :, 0], forecastsbforward[1, :, 1], color=forecast_color, marker="o", s=size_points)
    row1[3].set_title("SB-forward")
    
    if task_name == "LV":
        for ax in row1:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([0.75, 5])
    if task_name == "GoM":
        for ax in row1:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-1.5, 1.])

    # Colorbar #1: training slices
    norm_train = mcolors.Normalize(vmin=0, vmax=n_slices)
    sm_train = cm.ScalarMappable(norm=norm_train, cmap=cmap_slices)
    sm_train.set_array([])
    cb1 = fig.colorbar(sm_train, cax=cbar_ax_1)
    cb1.set_label("Time")

    # Legend for row1
    import matplotlib.lines as mlines
    train_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                 markersize=8, label='Training')
    forecast_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=8, label='Forecast')
    fig.legend(handles=[train_marker, forecast_marker],
               loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)

    # Compute common arrow scale for Row 2 and Row 3:
    all_vectors_2d = np.concatenate([
        gt_vector, model_vector, sbirr_vector, sbforward_vector,
        (model_vector - gt_vector), (sbirr_vector - gt_vector), (sbforward_vector - gt_vector)
    ], axis=0)
    max_len_2d = np.max(np.linalg.norm(all_vectors_2d, axis=-1))
    arrow_scale_2d = 1.0 * max_len_2d

    # Row 2: Actual vector fields (black quiver)
    row2[0].quiver(xx, yy, gt_vector[:, 0], gt_vector[:, 1],
                   color='black', width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    row2[1].quiver(xx, yy, model_vector[:, 0], model_vector[:, 1],
                   color='black', width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    row2[2].quiver(xx, yy, sbirr_vector[:, 0], sbirr_vector[:, 1],
                   color='black', width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    row2[3].quiver(xx, yy, sbforward_vector[:, 0], sbforward_vector[:, 1],
                   color='black', width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    
    if task_name == "LV":
        for ax in row2:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([0.75, 5])
    if task_name == "GoM":
        for ax in row2:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-1.5, 1.])

    # Row 3: Difference vector fields = (v_method - v_GT) colored by magnitude.
    # Compute differences:
    diff_mmd = model_vector - gt_vector
    diff_sbirr = sbirr_vector - gt_vector
    diff_sbforward = sbforward_vector - gt_vector
    # (We omit diff_gt as it is zero.)
    # Compute magnitudes and define normalization:
    mag_mmd = np.sqrt(np.sum(diff_mmd**2, axis=-1))
    mag_sbirr = np.sqrt(np.sum(diff_sbirr**2, axis=-1))
    mag_sbforward = np.sqrt(np.sum(diff_sbforward**2, axis=-1))
    all_mags = np.concatenate([mag_mmd, mag_sbirr, mag_sbforward])
    norm_diff = mcolors.Normalize(vmin=all_mags.min(), vmax=all_mags.max())

    # For 2D quiver, we can pass the magnitude array (C parameter) to get colored arrows.
    q0 = row3[0].quiver(xx, yy, (gt_vector-gt_vector)[:, 0], (gt_vector-gt_vector)[:, 1],
                          np.sqrt(np.sum((gt_vector-gt_vector)**2, axis=-1)),
                          cmap="plasma", norm=norm_diff,
                          width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    q1 = row3[1].quiver(xx, yy, diff_mmd[:, 0], diff_mmd[:, 1],
                          np.sqrt(np.sum(diff_mmd**2, axis=-1)),
                          cmap="plasma", norm=norm_diff,
                          width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    q2 = row3[2].quiver(xx, yy, diff_sbirr[:, 0], diff_sbirr[:, 1],
                          np.sqrt(np.sum(diff_sbirr**2, axis=-1)),
                          cmap="plasma", norm=norm_diff,
                          width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    q3 = row3[3].quiver(xx, yy, diff_sbforward[:, 0], diff_sbforward[:, 1],
                          np.sqrt(np.sum(diff_sbforward**2, axis=-1)),
                          cmap="plasma", norm=norm_diff,
                          width=0.005, angles='xy', scale_units='xy', scale=arrow_scale_2d)
    if task_name == "LV":
        for ax in row3:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([0.75, 5])
    if task_name == "GoM":
        for ax in row3:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-1.5, 1.])

    # Colorbar for difference magnitude (Row 3)
    cb2 = fig.colorbar(q3, cax=cbar_ax_2)
    cb2.set_label(r"Diff Magnitude")

    # Label leftmost column
    if task_name == "LV":
        row1[0].set_ylabel("Predator")
        row2[0].set_ylabel("Predator")
        row3[0].set_ylabel("Predator")
        for ax in row3:
            ax.set_xlabel("Prey")
    if task_name == "GoM":
        row1[0].set_ylabel("Latitude")
        row2[0].set_ylabel("Latitude")
        row3[0].set_ylabel("Latitude")
        for ax in row3:
            ax.set_xlabel("Longitude")

    plt.savefig(f"./figs/{outname}", dpi=300)
    # plt.show()
    # plt.close(fig)


###############################################################################
# ===================== 3D: Forecasting Figure (no vector fields) =============
###############################################################################
def plot_forecasting_3d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname="Repressilator_forecasting.png", cmap_str="coolwarm", size_points=30):
    """
    Single-row, 4-column 3D figure:
      [Ground Truth | Ours | SBIRR-ref | SB-forward].
    No vector fields – only training scatter plus final forecast scatter.
    One colorbar for training slices is placed in its own Axes.
    A legend (x = training, o = forecast) is placed below.
    """
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.15, top=0.88, wspace=0.2)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.73])  # colorbar for training slices

    axs = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    cmap_slices = cm.get_cmap(cmap_str)
    n_slices = len(Xs)
    if "pbmc" not in task_name:
        for i, X in enumerate(Xs):
            color_i = cmap_slices(i / (n_slices + 1))
            for ax_ in axs:
                ax_.scatter(X[:, 0], X[:, 1], X[:, 2], color=color_i, marker="x", s=size_points)
    else:
        for i, X in enumerate(Xs):
            
            color_i = cmap_slices(i / (n_slices + 1))
            for ax_ in axs:
                ax_.scatter(X[:, 0], X[:, 1], X[:, 2], color=color_i, marker="x", s=size_points)
    forecast_color = cmap_slices(0.95)
    axs[0].scatter(X_val[:, 0], X_val[:, 1], X_val[:, 2],
                   color=forecast_color, marker="o", s=size_points)
    axs[0].set_title("Ground Truth")
    axs[1].scatter(forecast[1, :, 0], forecast[1, :, 1], forecast[1, :, 2],
                   color=forecast_color, marker="o", s=size_points)
    axs[1].set_title("Ours")
    axs[2].scatter(forecastsbirr[1, :, 0], forecastsbirr[1, :, 1], forecastsbirr[1, :, 2],
                   color=forecast_color, marker="o", s=size_points)
    axs[2].set_title("SBIRR-ref")
    axs[3].scatter(forecastsbforward[1, :, 0], forecastsbforward[1, :, 1], forecastsbforward[1, :, 2],
                   color=forecast_color, marker="o", s=size_points)
    axs[3].set_title("SB-forward")

    if "pbmc" in task_name:
        for ax_ in axs:
            ax_.set_xticks([])
            ax_.set_yticks([])
            ax_.set_zticks([])
            ax_.set_xlabel("PC 1", labelpad=-3)
            ax_.set_ylabel("PC 2", labelpad=-3)
            ax_.set_zlabel("PC 3", labelpad=-3)
            '''
            ax_.set_xlim(1.5, 3.25)   
            ax_.set_ylim(-.5, .6)   
            ax_.set_zlim(0.75, 2) 
            ax_.view_init(-140, -60)
            '''
            ax_.set_xlim(-.3, 2)   
            ax_.set_ylim(-.5, .75)   
            ax_.set_zlim(0.5, 2) 
            ax_.view_init(-140, -140)
            
            #ax_.view_init(90, -90)
    else:
        for ax_ in axs:
            ax_.set_xticks([])
            ax_.set_yticks([])
            ax_.set_zticks([])
            ax_.set_xlabel("Gene 1", labelpad=-3)
            ax_.set_ylabel("Gene 2", labelpad=-3)
            ax_.set_zlabel("Gene 3", labelpad=-3)
            ax_.view_init(-140, 60)
    


    norm_train = mcolors.Normalize(vmin=0, vmax=n_slices)
    sm_train = cm.ScalarMappable(norm=norm_train, cmap=cmap_slices)
    sm_train.set_array([])
    cbar = fig.colorbar(sm_train, cax=cbar_ax)
    cbar.set_label("Time")

    import matplotlib.lines as mlines
    train_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                 markersize=8, label='Training')
    forecast_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=8, label='Forecast')
    fig.legend(handles=[train_marker, forecast_marker],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)

    plt.savefig(f"./figs/{outname}", dpi=300)
    # plt.show()
    # plt.close(fig)


###############################################################################
# ===================== 3D: Appendix Figure (3 rows: forecast, VF, diff VF) =====
###############################################################################
def plot_appendix_3d_diff_arrows(task_name, Xs, X_val,
                                 forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, zz, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname="Repressilator_appendix_diff_arrows.png",
                                 cmap_str="coolwarm", size_points=30):
    """
    3-row, 4-col 3D figure:
      Row 1: Forecasting scatter (training data & forecast) – uses one colormap.
      Row 2: Actual vector fields (black arrows) – uncolored.
      Row 3: Difference vector fields = (v_method - v_GT), where each arrow is colored
             according to its magnitude (using a second colormap).
    The same arrow scale is used for both Row 2 and Row 3.
    Two separate colorbars (one for Row 1 and one for the difference magnitude in Row 3)
    are added in separate Axes; each colorbar is sized to be 66% of the row height and aligned
    with the respective row.
    """
    fig = plt.figure(figsize=(24, 14))
    # Subplots region: leaving space on right for two separate colorbars.
    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.12, top=0.88, wspace=0.06, hspace=0.124)
    # For a 3-row grid, assume each row ~0.253 in height.
    # For 66% of each row, we want ~0.167 height for each colorbar.
    # Place one for Row 1 (top row) and one for Row 3 (bottom row).
    cbar_ax_1 = fig.add_axes([0.88, 0.67, 0.02, 0.167])  # for Row 1 (forecast training slices)
    cbar_ax_2 = fig.add_axes([0.88, 0.163, 0.02, 0.167]) # for Row 3 (difference magnitude)

    # Create subplots for each row.
    row1 = [fig.add_subplot(3, 4, i+1, projection='3d') for i in range(4)]
    row2 = [fig.add_subplot(3, 4, i+5, projection='3d') for i in range(4)]
    row3 = [fig.add_subplot(3, 4, i+9, projection='3d') for i in range(4)]

    # Use different colormaps:
    cmap_slices = cm.get_cmap("coolwarm")  # for Row 1
    cmap_diff = cm.get_cmap("plasma")        # for Row 3

    n_slices = len(Xs)

    for ax in row1:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("Gene 1")
        ax.set_ylabel("Gene 2")
        ax.set_zlabel("Gene 3")
        ax.view_init(-140, 60)
        ax.set_ylim([0.75, 5])

    # Row 1: Forecasting scatter
    for i, X in enumerate(Xs):
        color_i = cmap_slices(i / (n_slices + 1))
        for ax_ in row1:
            ax_.scatter(X[:, 0], X[:, 1], X[:, 2], color=color_i, marker="x", s=30)
    forecast_color = cmap_slices(0.95)
    row1[0].scatter(X_val[:, 0], X_val[:, 1], X_val[:, 2],
                    color=forecast_color, marker="o", s=size_points)
    row1[0].set_title("Ground Truth")
    row1[1].scatter(forecast[1, :, 0], forecast[1, :, 1], forecast[1, :, 2],
                    color=forecast_color, marker="o", s=size_points)
    row1[1].set_title("Ours")
    row1[2].scatter(forecastsbirr[1, :, 0], forecastsbirr[1, :, 1], forecastsbirr[1, :, 2],
                    color=forecast_color, marker="o", s=size_points)
    row1[2].set_title("SBIRR-ref")
    row1[3].scatter(forecastsbforward[1, :, 0], forecastsbforward[1, :, 1], forecastsbforward[1, :, 2],
                    color=forecast_color, marker="o", s=size_points)
    row1[3].set_title("SB-forward")
    
    # Colorbar for Row 1 (training slices)
    norm_train = mcolors.Normalize(vmin=0, vmax=n_slices)
    sm_train = cm.ScalarMappable(norm=norm_train, cmap=cmap_slices)
    sm_train.set_array([])
    cb1 = fig.colorbar(sm_train, cax=cbar_ax_1)
    cb1.set_label("Time")

    # Legend for Row 1
    import matplotlib.lines as mlines
    train_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                 markersize=8, label='Training')
    forecast_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=8, label='Forecast')
    fig.legend(handles=[train_marker, forecast_marker],
               loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=2)

    # Compute common arrow scale for Rows 2 and 3:
    all_row2 = np.concatenate([gt_vector, model_vector, sbirr_vector, sbforward_vector], axis=0)
    all_row3 = np.concatenate([model_vector - gt_vector, sbirr_vector - gt_vector, sbforward_vector - gt_vector], axis=0)
    all_vectors_3d = np.concatenate([all_row2, all_row3], axis=0)
    max_len_3d = np.max(np.linalg.norm(all_vectors_3d, axis=-1))
    arrow_scale_3d = 0.005 * max_len_3d

    # Helper function to plot colored arrows in 3D (one arrow at a time)
    def colored_quiver_3d(ax, X, Y, Z, U, V, W, cmap, norm, scale):
        for i in range(len(X)):
            mag = np.sqrt(U[i]**2 + V[i]**2 + W[i]**2)
            color = cmap(norm(mag))
            ax.quiver(X[i], Y[i], Z[i], U[i]*scale, V[i]*scale, W[i]*scale,
                      color=color, length=1.0, normalize=False, lw=0.8)

    # Row 2: Actual vector fields (black arrows)
            
    for ax in row2:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("Gene 1")
        ax.set_ylabel("Gene 2")
        ax.set_zlabel("Gene 3")
        ax.view_init(-140, 60)
        ax.set_ylim([0.75, 5])

    def black_quiver_3d(ax, X, Y, Z, U, V, W, scale):
        ax.quiver(X, Y, Z, U*scale, V*scale, W*scale, color='black', length=1.0, normalize=False, lw=0.8)

    black_quiver_3d(row2[0], xx, yy, zz, gt_vector[:, 0], gt_vector[:, 1], gt_vector[:, 2], arrow_scale_3d)
    black_quiver_3d(row2[1], xx, yy, zz, model_vector[:, 0], model_vector[:, 1], model_vector[:, 2], arrow_scale_3d)
    black_quiver_3d(row2[2], xx, yy, zz, sbirr_vector[:, 0], sbirr_vector[:, 1], sbirr_vector[:, 2], arrow_scale_3d)
    black_quiver_3d(row2[3], xx, yy, zz, sbforward_vector[:, 0], sbforward_vector[:, 1], sbforward_vector[:, 2], arrow_scale_3d)
    

    # Row 3: Difference arrows, colored by magnitude using cmap_diff.
    diff_mmd = model_vector - gt_vector
    diff_sbirr = sbirr_vector - gt_vector
    diff_sbforward = sbforward_vector - gt_vector
    # For a common norm, compute magnitudes over all differences:
    mag_mmd = np.sqrt(np.sum(diff_mmd**2, axis=-1))
    mag_sbirr = np.sqrt(np.sum(diff_sbirr**2, axis=-1))
    mag_sbforward = np.sqrt(np.sum(diff_sbforward**2, axis=-1))
    all_mags = np.concatenate([mag_mmd, mag_sbirr, mag_sbforward])
    norm_diff = mcolors.Normalize(vmin=all_mags.min(), vmax=all_mags.max())

    for ax in row3:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("Gene 1")
        ax.set_ylabel("Gene 2")
        ax.set_zlabel("Gene 3")
        ax.view_init(-140, 60)
        ax.set_ylim([0.75, 5])

    # Use the helper to plot colored arrows (one arrow at a time)
    colored_quiver_3d(row3[0], xx, yy, zz, (gt_vector-gt_vector)[:, 0], (gt_vector-gt_vector)[:, 1], (gt_vector-gt_vector)[:, 2],
                      cmap_diff, norm_diff, arrow_scale_3d)
    colored_quiver_3d(row3[1], xx, yy, zz, diff_mmd[:, 0], diff_mmd[:, 1], diff_mmd[:, 2],
                      cmap_diff, norm_diff, arrow_scale_3d)
    colored_quiver_3d(row3[2], xx, yy, zz, diff_sbirr[:, 0], diff_sbirr[:, 1], diff_sbirr[:, 2],
                      cmap_diff, norm_diff, arrow_scale_3d)
    colored_quiver_3d(row3[3], xx, yy, zz, diff_sbforward[:, 0], diff_sbforward[:, 1], diff_sbforward[:, 2],
                      cmap_diff, norm_diff, arrow_scale_3d)
    
    # Colorbar for Row 3 (difference magnitude)
    sm_diff = cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff)
    sm_diff.set_array([])
    cb2 = fig.colorbar(sm_diff, cax=cbar_ax_2)
    cb2.set_label(r"Diff Magnitude")

    plt.savefig(f"./figs/{outname}", dpi=300)
    # plt.show()
    # plt.close(fig)


###############################################################################
# =============================== Main ========================================
###############################################################################
if __name__ == "__main__":
    seed_use = 42
    
    # -------------------------- Lotka-Volterra -------------------------
    task_name = "LV"
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use)

    xx = vector_fields['xx']
    yy = vector_fields['yy']
    gt_vector = vector_fields['gt_vector']
    model_vector = vector_fields['model_vector']
    sbirr_vector = vector_fields['sbirr_vector']
    sbforward_vector = vector_fields['sbforward_vector']

    # 2D Forecasting
    plot_forecasting_2d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname="LV_forecasting.pdf", cmap_str="coolwarm", size_points=20)

    # 2D Appendix with difference magnitude coloring in Row 3
    plot_appendix_2d_diff_arrows(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname="LV_appendix_diff_arrows.pdf", size_points=20)

    # -------------------------- Repressilator -------------------------
    task_name = "Repressilator"
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use)

    xx = vector_fields['xx']
    yy = vector_fields['yy']
    zz = vector_fields['zz']
    gt_vector = vector_fields['gt_vector']
    model_vector = vector_fields['model_vector']
    sbirr_vector = vector_fields['sbirr_vector']
    sbforward_vector = vector_fields['sbforward_vector']

    # 3D Forecasting
    plot_forecasting_3d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname="Repressilator_forecasting.pdf", cmap_str="coolwarm", size_points=10)

    # 3D Appendix with difference magnitude coloring
    plot_appendix_3d_diff_arrows(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, zz, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname="Repressilator_appendix_diff_arrows.pdf", cmap_str="coolwarm", size_points=10)

    # -------------------------- Repressilator MLP -------------------------
    task_name = "Repressilator"
    kind = "mlp"
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use, kind)
    
    xx = vector_fields['xx']
    yy = vector_fields['yy']
    zz = vector_fields['zz']
    gt_vector = vector_fields['gt_vector']
    model_vector = vector_fields['model_vector']
    sbirr_vector = vector_fields['sbirr_vector']
    sbforward_vector = vector_fields['sbforward_vector']

    # 3D Forecasting
    plot_forecasting_3d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname=f"{kind}_Repressilator_forecasting.pdf", cmap_str="coolwarm", size_points=10)

    # 3D Appendix for MLP
    plot_appendix_3d_diff_arrows(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, zz, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname=f"{kind}_Repressilator_appendix_diff_arrows.pdf", cmap_str="coolwarm", size_points=10)

    # -------------------------- Repressilator missing obs -------------------------
    task_name = "Repressilator"
    kind = "missingobs"
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use, kind)
    
    # 3D Forecasting for missing obs
    plot_forecasting_3d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname=f"{kind}_Repressilator_forecasting.pdf", cmap_str="coolwarm", size_points=10)
    
    # -------------------------- GoM -----------------------------------------------
    task_name = "GoM"
    kind = "realdata"
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use, kind)

    xx = vector_fields['xx']
    yy = vector_fields['yy']
    gt_vector = vector_fields['gt_vector']
    model_vector = vector_fields['model_vector']
    sbirr_vector = vector_fields['sbirr_vector']
    sbforward_vector = vector_fields['sbforward_vector']

    # 2D Forecasting
    plot_forecasting_2d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname="GoM_forecasting.pdf", cmap_str="coolwarm", size_points=20)

    # 2D Appendix with difference magnitude coloring in Row 3
    plot_appendix_2d_diff_arrows(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                                 xx, yy, gt_vector, model_vector, sbirr_vector, sbforward_vector,
                                 outname="GoM_appendix_diff_arrows.pdf", size_points=20)
    
    # -------------------------- pbmc -----------------------------------------------
    task_name = "pbmc"
    kind = "realdata"
    #breakpoint()
    (Xs, X_val, forecast, forecastsbirr, forecastsbforward,
     model, sbirr, sbforward, time_scale, vector_fields) = get_data(task_name, seed_use, kind)

    # 3D Forecasting for missing obs
    plot_forecasting_3d(task_name, Xs, X_val, forecast, forecastsbirr, forecastsbforward,
                        outname=f"pbmc_forecasting.pdf", cmap_str="coolwarm", size_points=10)