"""Regenerate the figures of the paper that are computed from forecasts and scores:
  - forecast figures (Figs. 1-4 and the LV figure of App. D): training snapshots, truth and the forecasts of SnapMMD, SBIRR-ref
    and SB-forward for seed 42, with the plotting functions used for the paper (code/plotting/);
  - vector-field difference figures of App. D (LV, the two repressilator families, GoM), from outputs/vector_fields (seed 42);
  - interpolation-metric figures of App. D (MMD^2 and EMD at each validation time, 10 methods), from results/scores.csv;
  - galleries of every method of the appendix tables (App. E): forecasts (seed 42) from outputs/ and generated/, and interpolation
    trajectories (seed 44) from outputs/paths_for_figures and generated/paths_for_figures (written by simulate.py);
    {stem}_forecasting_all.png and {stem}_interpolation_all.png.
PBMC interpolation grids of every method: pbmc_interpolation_grid_{1-4}{a,b}.png. The PBMC progression figure is not regenerated. Output: results/figures/. Requires a LaTeX installation (the plots use text.usetex)."""
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, DATA, TASKS

FIG = f"{ROOT}/results/figures"; os.makedirs(f"{FIG}/figs", exist_ok=True); os.chdir(FIG)   # the plotting functions save to ./figs/
sys.path.insert(0, f"{ROOT}/code/plotting")
import make_plot as mp                      # noqa: E402  (the figure functions used for the paper, unchanged)
import pbmc_forecasting_maintext as pf      # noqa: E402

SEED = 42


def fc(method, label):
    root = "generated" if method == "Ours" else "outputs"
    x = np.load(f"{ROOT}/{root}/{method}/{label}/seed_{SEED}.npz")["forecast"]
    return np.stack([x, x])                 # the plotting functions read forecast[1]


def data(label):
    d = np.load(f"{DATA}/{TASKS[label]}.npz"); Xs = [d["Xs"][i] for i in range(int(d["N_steps"]) - 1)]
    return Xs, d["Xs"][-1]


for label, name, plot in [("LV", "LV", "2d"), ("GoM", "GoM", "2d"), ("ReprParam", "Repressilator", "3d"), ("ReprSemiparam", "mlp_Repressilator", "3d"),
                          ("ReprProtein", "missingobs_Repressilator", "3d")]:
    Xs, Xv = data(label); args = (name.split("_")[-1], Xs, Xv, fc("Ours", label), fc("SBIRR-ref", label), fc("SB-forward", label))
    (mp.plot_forecasting_2d if plot == "2d" else mp.plot_forecasting_3d)(*args, outname=f"{name}_forecasting.pdf")
    vf = {"LV": "classic/LV", "GoM": "realdata/GoM", "ReprParam": "classic/Repressilator", "ReprSemiparam": "mlp/Repressilator"}.get(label)
    if vf:
        v = np.load(f"{ROOT}/outputs/vector_fields/{vf.split('/')[0]}/{vf.split('/')[1]}_vector_field_{SEED}.npz")
        grid = [v["xx"], v["yy"]] + ([v["zz"]] if plot == "3d" else [])
        fn = mp.plot_appendix_2d_diff_arrows if plot == "2d" else mp.plot_appendix_3d_diff_arrows
        fn(*args, *grid, v["gt_vector"], v["model_vector"], v["sbirr_vector"], v["sbforward_vector"], outname=f"{name}_appendix_diff_arrows.pdf")
    print(label, "forecast figure done", flush=True)
Xs, Xv = data("PBMC")
pf.plot_pbmc_forecasting_maintext(Xs, Xv, fc("Ours", "PBMC"), fc("SBIRR-ref", "PBMC"), fc("SB-forward", "PBMC"), outname="pbmc_forecasting_maintext.pdf")

# galleries of every method in the appendix tables (App. E): forecasts (seed 42) and interpolation trajectories (seed 44)
import galleries as gal                     # noqa: E402
from common import Task, FIG_SEED_FORECAST, FIG_SEED_INTERP   # noqa: E402
GENERATED = {"Ours", "Ours (mRNA-only model)", "Persistence", "OT midpoint"}      # written by simulate.py and sanity.py
SHOW = {"PI-SDE (learned sigma)": r"PI-SDE (learned $\sigma$)", "Ours (mRNA-only model)": "Ours (mRNA-only model)"}
FC_METHODS = ["Ours", "Ours (mRNA-only model)", "SBIRR-ref", "SB-forward", "PRESCIENT", "PI-SDE", "PI-SDE (learned sigma)", "scNODE", "JKOnet*",
              "JKOnet* (full)", "Persistence", "SBIRR-ref (last snapshot)", "SB-forward (last snapshot)"]
IN_METHODS = ["Ours", "SBIRR", "DMSB", "OT-CFM", "SB-CFM", "SF2M", "PRESCIENT", "PI-SDE", "PI-SDE (learned sigma)", "scNODE", "JKOnet*",
              "JKOnet* (full)", "Persistence", "OT midpoint"]
STEMS = {"LV": "LV", "ReprParam": "Repressilator", "ReprSemiparam": "mlp_Repressilator", "ReprProtein": "missingobs_Repressilator",
         "GoM": "GoM", "PBMC": "pbmc"}


def output(method, label, seed):
    simple = method in ("Persistence", "OT midpoint")
    f = f"{ROOT}/{'generated' if method in GENERATED else 'outputs'}/{method}/{label}/seed_{0 if simple else seed}.npz"
    return np.load(f) if os.path.exists(f) else None


for label, stem in STEMS.items():
    t = Task(label); rng = np.random.default_rng(0)
    panels = [(SHOW.get(m, m), o["forecast"]) for m in FC_METHODS if (o := output(m, label, FIG_SEED_FORECAST)) is not None and "forecast" in o]
    gal.forecast_gallery(label, list(t.train), t.forecast_truth, panels, f"{FIG}/figs/{stem}_forecasting_all.png", show_train=label != "PBMC")
    panels = []
    for m in IN_METHODS:
        if m in ("Persistence", "OT midpoint"):
            x = output(m, label, 0)["interp"]; panels.append((m, "points", x[:, rng.choice(x.shape[1], min(200, x.shape[1]), replace=False)]))
        else:
            p = f"{ROOT}/{'generated' if m in GENERATED else 'outputs'}/paths_for_figures/{m}/{label}.npz"
            if os.path.exists(p):
                panels.append((SHOW.get(m, m), "path", np.load(p)["path"]))
    gal.interpolation_gallery(label, list(t.val_truth), panels, f"{FIG}/figs/{stem}_interpolation_all.png",
                              max_val_points=100 if label == "PBMC" else None)
    print(label, "galleries done", flush=True)

# PBMC: particles of every interpolation method at the validation times, five times per block (four in the last), methods split over two figures
t = Task("PBMC"); hours = [0.5 + k for k in range(t.n_val)]
groups = {"a": ["Ours", "SBIRR", "DMSB", "OT-CFM", "SB-CFM", "SF2M", "PRESCIENT"],
          "b": ["PI-SDE", "PI-SDE (learned sigma)", "scNODE", "JKOnet*", "JKOnet* (full)", "Persistence", "OT midpoint"]}
for block in range(4):
    times = [(k, f"{hours[k]:g} h") for k in range(5 * block, min(5 * block + 5, t.n_val))]
    for g, methods in groups.items():
        rows = [(SHOW.get(m, m), o["interp"]) for m in methods
                if (o := output(m, "PBMC", 0 if m in ("Persistence", "OT midpoint") else FIG_SEED_INTERP)) is not None and "interp" in o]
        gal.interpolation_grid("PBMC", list(t.val_truth), rows, times, f"{FIG}/figs/pbmc_interpolation_grid_{block + 1}{g}.png")
print("PBMC interpolation grids done", flush=True)

# interpolation-metric figures (as in the paper: MMD^2 left, EMD right, mean +- SD over seeds, EMD axis capped at 5)
S = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f"{ROOT}/results/scores.csv")):
    if r["kind"] == "interp":
        S[(r["method"], r["task"])][("mmd2", int(r["t"]))].append(float(r["mmd2"])); S[(r["method"], r["task"])][("emd", int(r["t"]))].append(float(r["emd"]))
METHODS = ["Ours", "SBIRR", "DMSB", "OT-CFM", "SB-CFM", "SF2M", "PRESCIENT", "PI-SDE", "scNODE", "JKOnet*"]
colors = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#009E73", "#000000", "#999999", "#8B4513"]
styles = ["-", "--", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), "--", ":", (0, (5, 1)), "-."]
FILES = {"LV": "LV_classic", "ReprParam": "Repressilator_classic", "ReprSemiparam": "Repressilator_mlp", "ReprProtein": "Repressilator_missingobs",
         "GoM": "GoM_realdata", "PBMC": "pbmc_realdata"}
matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 24})
for label, f in FILES.items():
    n_val = max(t for (_, t) in S[("Ours", label)]) + 1; steps = np.arange(n_val) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, met, title in [(axes[0], "mmd2", r"MMD$^2$ on interpolation"), (axes[1], "emd", "EMD on interpolation")]:
        top = 0
        for i, m in enumerate(METHODS):
            if not S[(m, label)]: continue
            mu = np.array([np.mean(S[(m, label)][(met, t)]) for t in range(n_val)]); sd = np.array([np.std(S[(m, label)][(met, t)]) for t in range(n_val)])
            ax.plot(steps, mu, label=m if met == "mmd2" else None, color=colors[i], linestyle=styles[i], linewidth=2.5 if m == "Ours" and met == "mmd2" else 1.5)
            ax.fill_between(steps, mu - sd, mu + sd, color=colors[i], alpha=0.30); top = max(top, (mu + sd).max())
        ax.set_title(title); ax.set_xlabel("Validation time"); ax.set_xticks(steps[::2] if n_val > 12 else steps)
        ax.tick_params(axis="x", labelrotation=45); ax.grid(ls=":", alpha=0.4); ax.set_ylim(0, min(1.05 * top, 5))
    fig.subplots_adjust(left=0.25); fig.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), frameon=False)
    fig.savefig(f"{FIG}/figs/{f}_interpolation_metric.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written to", f"{FIG}/figs")
