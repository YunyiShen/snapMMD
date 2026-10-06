"""Score every method's outputs (outputs/ and generated/) against the held-out snapshots. Writes results/scores.csv:
method, task, seed, kind (forecast / interp), t (validation index; -1 for the forecast), mmd2 for the bandwidths 'snap' (the paper's
metric), 'snap/4', '4snap', 'h_train', and emd. Methods without a model family appear under both ReprParam and ReprSemiparam."""
import csv, glob, os, sys, time
import numpy as np
from common import ROOT, TASKS, Task, mmd2, emd

rows = []; t0 = time.time(); tasks = {l: Task(l) for l in TASKS}
files = sorted(glob.glob(f"{ROOT}/outputs/*/*/seed_*.npz") + glob.glob(f"{ROOT}/generated/*/*/seed_*.npz"))
only = sys.argv[1:]                                    # optional method names to score (default: all)
for f in files:
    method, label = f.split(os.sep)[-3], f.split(os.sep)[-2]; seed = int(os.path.basename(f)[5:-4])
    if only and method not in only: continue
    t = tasks[label]; z = np.load(f)
    if "forecast" in z.files and np.isfinite(z["forecast"]).all():
        m = mmd2(t.forecast_truth, z["forecast"], t.h_train); rows.append([method, label, seed, "forecast", -1, m["snap"], m["snap/4"], m["4snap"], m["h_train"], emd(z["forecast"], t.forecast_truth)])
    if "interp" in z.files and np.isfinite(z["interp"]).all():
        for i in range(t.n_val):
            Y = z["interp"][i]; m = mmd2(t.val_truth[i], Y, t.h_train)
            rows.append([method, label, seed, "interp", i, m["snap"], m["snap/4"], m["4snap"], m["h_train"], emd(Y, t.val_truth[i])])
    print(f"{method:28s} {label:14s} seed {seed:2d} ({time.time() - t0:.0f}s)", flush=True)
os.makedirs(f"{ROOT}/results", exist_ok=True)
out = f"{ROOT}/results/scores.csv"
old = [r for r in csv.reader(open(out))][1:] if (only and os.path.exists(out)) else []
old = [r for r in old if r[0] not in only]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["method", "task", "seed", "kind", "t", "mmd2", "mmd2_snap/4", "mmd2_4snap", "mmd2_h_train", "emd"]); w.writerows(old + rows)
print("wrote", out, len(old) + len(rows), "rows")
