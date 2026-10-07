"""Galleries of every method (App. E): forecasts (seed 42) and interpolation trajectories (seed 44), in the style of the
paper's forecast and interpolation figures (make_plot.py).

Every panel of a task uses the same axes, fixed to the range of the data (training, validation and held-out snapshots) with a
margin, so that a method whose particles leave the data does not rescale its panel. Points and trajectory segments outside
the axes are not drawn; the panel title gives the share of the method's particles outside (forecasts) or of its trajectories
that leave the axes at some time (interpolation), when it is at least 1%.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib import cm

NCOL = 4
# task -> (axis labels, fixed limits or None for data-driven, 3D view or None for 2D)
STYLE = {
    "LV": (("Prey", "Predator"), [(None, None), (0.75, 5)], None),
    "GoM": (("Longitude", "Latitude"), [(-1.5, 1.0), (-1.5, 1.1)], None),
    "ReprParam": (("Gene 1", "Gene 2", "Gene 3"), None, (-140, 60)),
    "ReprSemiparam": (("Gene 1", "Gene 2", "Gene 3"), None, (-140, 60)),
    "ReprProtein": (("Gene 1", "Gene 2", "Gene 3"), None, (-140, 60)),
    "PBMC": (("Program 1", "Program 2", "Program 3"), None, (-140, -140)),   # view of the main-text PBMC figure; first three gene programs
}


def limits(task, clouds, margin=0.3):
    """Axis limits: the task's fixed limits where the paper uses them, otherwise the data range plus a margin."""
    allx = np.concatenate([c.reshape(-1, c.shape[-1])[:, :3] for c in clouds])          # the plotted coordinates
    lo, hi = allx.min(0), allx.max(0); pad = margin * (hi - lo)
    lims = [(lo[k] - pad[k], hi[k] + pad[k]) for k in range(allx.shape[1])]
    fixed = STYLE[task][1]
    if fixed:
        lims = [(f[0] if f[0] is not None else l[0], f[1] if f[1] is not None else l[1]) for f, l in zip(fixed, lims)]
    return lims


def inside(X, lims):
    return np.all([(X[..., k] >= a) & (X[..., k] <= b) for k, (a, b) in enumerate(lims)], axis=0)


def _axes(fig, task, n, lims):
    nrow = int(np.ceil(n / NCOL)); three = STYLE[task][2] is not None; labels = STYLE[task][0]
    axs = []
    for i in range(n):
        ax = fig.add_subplot(nrow, NCOL, i + 1, projection="3d" if three else None)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(labels[0], labelpad=-8 if three else 4); ax.set_ylabel(labels[1], labelpad=-8 if three else 4)
        if three:
            ax.set_zlim(lims[2]); ax.set_zticks([]); ax.set_zlabel(labels[2], labelpad=-8); ax.view_init(*STYLE[task][2])
            ax.computed_zorder = False                              # draw in call order: trajectories over the markers
        axs.append(ax)
    return axs, three


def _scatter(ax, X, three, **kw):
    if len(X):
        ax.scatter(*(X[:, k] for k in range(3 if three else 2)), **kw)


def _title(ax, name, share, phrase):
    pct = "\\%" if plt.rcParams["text.usetex"] else "%"
    ax.set_title(name if share < 0.01 else f"{name}\n({100 * share:.0f}{pct} {phrase})", fontsize=20 if share < 0.01 else 17)


def _finish(fig, n_slices, cbar_label, handles, outname):
    if cbar_label:
        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=n_slices), cmap=cm.get_cmap("coolwarm")); sm.set_array([])
        fig.colorbar(sm, cax=cax).set_label(cbar_label)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.06, top=0.95, wspace=0.15, hspace=0.3)
    fig.savefig(outname, dpi=150); plt.close(fig)


def forecast_gallery(task, train, truth, panels, outname, size=12, show_train=True):
    """train: list of (N, d) training snapshots; truth: (N, d) held-out snapshot; panels: list of (name, (N, d) forecast).
    show_train=False leaves the training snapshots out of the panels (PBMC, as in the main-text PBMC figure)."""
    cmap = cm.get_cmap("coolwarm"); fc_color = cmap(0.95)
    lims = limits(task, list(train) + [truth])
    allp = [("Ground Truth", truth)] + panels
    nrow = int(np.ceil(len(allp) / NCOL))
    fig = plt.figure(figsize=(5 * NCOL, 4.6 * nrow)); axs, three = _axes(fig, task, len(allp), lims)
    for ax, (name, X) in zip(axs, allp):
        for i, S in enumerate(train if show_train else []):
            _scatter(ax, S, three, color=cmap(i / (len(train) + 1)), marker="x", s=size, linewidths=0.6)
        ok = inside(X, lims); _scatter(ax, X[ok], three, color=fc_color, marker="o", s=size)
        _title(ax, name, 1 - ok.mean(), "of particles outside")
    handles = ([mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=8, label="Training")] if show_train else []) + \
              [mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Forecast")]
    _finish(fig, len(train), "Time" if show_train else None, handles, outname)


def interpolation_gallery(task, val, panels, outname, size=12, max_val_points=None):
    """val: list of (N, d) validation snapshots; panels: list of (name, kind, data) with kind 'path' and data (N, T, d) a
    simulated path, or kind 'points' and data (n_val, N, d) the method's interpolants at the validation times.
    max_val_points: draw at most this many points of each validation snapshot (PBMC: 100 of 500, for legibility)."""
    if max_val_points:
        rng = np.random.default_rng(0); val = [S[rng.choice(len(S), min(max_val_points, len(S)), replace=False)] for S in val]
    cmap = cm.get_cmap("coolwarm")
    lims = limits(task, list(val))
    nrow = int(np.ceil(len(panels) / NCOL))
    fig = plt.figure(figsize=(5 * NCOL, 4.6 * nrow)); axs, three = _axes(fig, task, len(panels), lims)
    for ax, (name, kind, X) in zip(axs, panels):
        for i, S in enumerate(val):
            _scatter(ax, S, three, color=cmap(i / (len(val) + 1)), marker="x", s=size, linewidths=0.6)
        if kind == "path" and np.abs(X[:, -1] - X[:, 0]).max() < 1e-5:           # the particles do not move: draw them as points
            ok = inside(X[:, 0], lims); _scatter(ax, X[ok, 0], three, color="black", marker="o", s=3, alpha=0.4)
            ax.set_title(f"{name}\n(particles do not move)", fontsize=17)
        elif kind == "path":
            ok = inside(X, lims); P = np.where(ok[..., None], X, np.nan)        # segments outside the axes are not drawn
            for p in P:
                ax.plot(*(p[:, k] for k in range(3 if three else 2)), color="black", alpha=0.1, lw=0.6)
            share = 1 - ok.all(axis=1).mean()
            _title(ax, name, share, "of trajectories leave the axes")
        else:
            pts = X.reshape(-1, X.shape[-1]); ok = inside(pts, lims)
            _scatter(ax, pts[ok], three, color="black", marker="o", s=3, alpha=0.4)
            _title(ax, name, 1 - ok.mean(), "of points outside")
    handles = [mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=8, label="Validation points"),
               mlines.Line2D([], [], color="black", label="Trajectories"),
               mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=4, label="Interpolants (simple baselines)")]
    _finish(fig, len(val), "Validation time", handles, outname)


def interpolation_grid(task, val, rows, times, outname, max_points=500, size=4):
    """Particles of each method at a few validation times (PBMC): rows = ground truth and the methods, columns = times.
    val: list of (N, d) validation snapshots; rows: list of (name, (n_val, N, d) interpolants); times: (index, label) pairs."""
    cmap = cm.get_cmap("coolwarm"); rng = np.random.default_rng(0)
    lims = limits(task, list(val)); view = STYLE[task][2]
    allr = [("Ground Truth", np.stack(val))] + rows
    fig = plt.figure(figsize=(4 * len(times), 3.6 * len(allr)))
    for i, (name, X) in enumerate(allr):
        shares = []
        for j, (t, tlabel) in enumerate(times):
            ax = fig.add_subplot(len(allr), len(times), i * len(times) + j + 1, projection="3d")
            P = X[t][:, :3]
            if len(P) > max_points:
                P = P[rng.choice(len(P), max_points, replace=False)]
            ok = inside(P, lims); shares.append(1 - ok.mean())
            ax.scatter(P[ok, 0], P[ok, 1], P[ok, 2], color=cmap(0.95) if i == 0 else cmap(0.7), marker="o" if i == 0 else "x", s=size)
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2]); ax.view_init(*view)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if i == 0:
                ax.set_title(tlabel, fontsize=18)
        share = float(np.mean(shares)); pct = "\\%" if plt.rcParams["text.usetex"] else "%"
        label = name if share < 0.01 else f"{name}\n({100 * share:.0f}{pct} outside)"
        fig.text(0.02, 1 - (i + 0.5) / len(allr), label, rotation=90, ha="center", va="center", fontsize=18)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.01, top=0.97, wspace=0.0, hspace=0.05)
    fig.savefig(outname, dpi=110); plt.close(fig)
