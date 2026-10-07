"""Campaign 4: score the new baselines with the paper's metric (Campaign 2) and EMD, next to the existing methods.

Reads results/outputs/{method}_{variant}/{task}_{forecast,interpolation}_{seed}.npz and writes
  results/scores.csv        one row per (method, variant, task, seed, task_type, val_index): mmd2 (snapshot bandwidth), emd
  results/summary.txt       per task: forecast and interpolation means next to SnapMMD, SBIRR-ref/SBIRR, SB-forward (Campaign 2 values)
Run from this folder: python score.py
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))                # repository root (data/ is here)
DATA = os.path.join(W, "data")
import sys

# data of each task (the truth at the forecast time and the validation snapshots), as in the paper's evaluation
STEMS = {"LV": "classic/LV_data", "ReprParam": "classic/Repressilator_data", "ReprSemiparam": "classic/Repressilator_data",
         "ReprProtein": "missingobs/Repressilator_data", "GoM": "realdata/GoM_data",
         "PBMC": "realdata/processed_pbmc_data_sub500_every_2_until20"}


class Task:
    def __init__(self, label):
        train = np.load(f"{DATA}/{STEMS[label]}.npz")
        val = np.load(f"{DATA}/{STEMS[label]}_interp_val.npz")
        self.label = label
        self.forecast_truth = train["Xs"][-1]
        self.val_truth = val["Xs"]
        self.d = train["Xs"].shape[-1]


def midpoints(n_steps, n_val):
    """Time-index rule of the paper's interpolation metric: middle of each of n_val equal segments of the simulated path."""
    pts = np.linspace(0, n_steps - 1, n_val + 1, dtype=int)
    return (pts[1:] + pts[:-1]) // 2
import ot
from sklearn.metrics.pairwise import pairwise_distances


def emd(p, q):
    """Same computation as TrajectoryNet's earth_mover_distance, i.e. the draft's EMD (2-Wasserstein)."""
    M = np.ascontiguousarray(pairwise_distances(p, Y=q, metric="sqeuclidean"))
    return np.sqrt(ot.emd2(np.ones(len(p)) / len(p), np.ones(len(q)) / len(q), M, numItermax=1e7))

from snapMMD.dls import evaluation_mmd                          # noqa: E402

# label used by run_baselines -> labels of the paper rows that share its data
ROWS = {"LV": ["LV"], "ReprParam": ["ReprParam", "ReprSemiparam"], "ReprProtein": ["ReprProtein"], "GoM": ["GoM"], "PBMC": ["PBMC"]}

tasks = {}
rows = []
for d in sorted(glob.glob(f"{HERE}/results/outputs/*")):
    mv = os.path.basename(d)
    if mv.endswith("_smoke"):
        continue
    method, variant = mv.split("_", 1)
    for f in sorted(glob.glob(f"{d}/*_forecast_*.npz")):
        m = re.match(r"(.+)_forecast_(\d+)\.npz", os.path.basename(f))
        label, seed = m.group(1), int(m.group(2))
        if label not in tasks:
            tasks[label] = Task(ROWS[label][0])
        tk = tasks[label]
        fc = np.load(f)["forecast"][-1][:, :tk.d]
        X = tk.forecast_truth
        rows.append([method, variant, label, seed, "forecast", -1,
                     evaluation_mmd(torch.tensor(X), torch.tensor(fc)).item(), emd(fc, X) if label != "PBMC" else float("nan")])
        tr = np.load(f.replace("_forecast_", "_interpolation_"))["interpolation"]
        # PBMC: paths on a 0.1 h grid over 0-19 h (191 points), read at the validation times 0.5, ..., 18.5 h;
        # 19.5 h, after the last training time, is not scored. Other tasks: the midpoint rule is exact.
        idx = np.arange(5, 190, 10) if label == "PBMC" else midpoints(tr.shape[0], tk.val_truth.shape[0])
        assert label != "PBMC" or tr.shape[0] == 191
        for i, j in enumerate(idx):
            Y, Xv = tr[j][:, :tk.d], tk.val_truth[i]
            rows.append([method, variant, label, seed, "interp", i,
                         evaluation_mmd(torch.tensor(Xv), torch.tensor(Y)).item(), emd(Y, Xv) if label != "PBMC" else float("nan")])
        # JKOnet* under the authors' one-step-ahead protocol (arrays saved by run_jkonet since 2026-10-04 09:30):
        # scored as the pseudo-variant '<variant>-1step'
        z = np.load(f)
        if method == "jkonet" and "forecast_onestep" in z.files and "interpolation_onestep" in z.files:
            fc1, tr1 = z["forecast_onestep"][:, :tk.d], z["interpolation_onestep"]
            rows.append([method, variant + "-1step", label, seed, "forecast", -1,
                         evaluation_mmd(torch.tensor(X), torch.tensor(fc1)).item(), emd(fc1, X) if label != "PBMC" else float("nan")])
            for i, j in enumerate(idx):
                Y, Xv = tr1[j][:, :tk.d], tk.val_truth[i]
                rows.append([method, variant + "-1step", label, seed, "interp", i,
                             evaluation_mmd(torch.tensor(Xv), torch.tensor(Y)).item(), emd(Y, Xv) if label != "PBMC" else float("nan")])
with open(f"{HERE}/results/scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "variant", "task", "seed", "task_type", "val_index", "mmd2", "emd"])
    w.writerows(rows)

# ---- summary next to the existing methods (Campaign 2, definition 'snap')
c2 = defaultdict(list)
C2 = os.environ.get("SNAPMMD_VALUES_CSV", "")                                     # the paper's own evaluation (optional)
for r in (csv.DictReader(open(C2)) if C2 and os.path.exists(C2) else []):
    if r["definition"] == "snap" and r["table"] in ("forecast", "interp"):
        c2[(r["table"], r["task"], r["method"])].append(float(r["value"]))
new = defaultdict(list)
for method, variant, label, seed, tt, vi, mmd2, e in rows:
    new[(tt, label, f"{method}:{variant}")].append(mmd2)
with open(f"{HERE}/results/summary.txt", "w") as out:
    for tt, title, old_methods in [("forecast", "FORECAST MMD^2 (mean over seeds)", ["Ours", "SBIRR-ref", "SB-forward"]),
                                   ("interp", "INTERPOLATION MMD^2 (mean over seeds x validation times)", ["Ours", "SBIRR", "DMSB", "OT-CFM", "SB-CFM", "SF2M"])]:
        out.write(f"\n== {title}\n")
        for label, paper_rows in ROWS.items():
            variants = sorted({k[2] for k in new if k[0] == tt and k[1] == label})
            if not variants:
                continue
            for prow in paper_rows:
                line = f"{prow:14s} " + " ".join(f"{m}={np.mean(c2[(tt, prow, m)]):.3f}" for m in old_methods if c2[(tt, prow, m)])
                line += " | " + " ".join(f"{v}={np.mean(new[(tt, label, v)]):.3f}(n={len(new[(tt, label, v)])})" for v in variants)
                out.write(line + "\n")
print(open(f"{HERE}/results/summary.txt").read())
