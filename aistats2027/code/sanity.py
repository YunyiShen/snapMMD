"""Simple baselines that need no fit (App. D.3): persistence (forecast: last training snapshot; interpolation: previous training
snapshot) and the OT midpoint (midpoint of the exact OT matching between neighbouring snapshots). Writes generated/{method}/{task}/seed_0.npz.
The third simple baseline, the SB references propagated from the last snapshot, is in outputs/ (regenerable from
checkpoints/schrodinger_bridge with the SBIRR package; see README)."""
import os
import numpy as np, ot
from common import ROOT, TASKS, Task, save_output


def ot_midpoint(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    M = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)
    P = ot.emd(np.ones(len(A)) / len(A), np.ones(len(B)) / len(B), np.ascontiguousarray(M), numItermax=int(1e7))
    j = P.argmax(1); assert len(set(j.tolist())) == len(A), "OT plan is not a matching"
    return 0.5 * (A + B[j])


for label in TASKS:
    t = Task(label); snaps = list(t.train) + ([t.forecast_truth] if label == "PBMC" else [])
    pers = np.stack([snaps[i] for i in range(t.n_val)]); mid = np.stack([ot_midpoint(snaps[i], snaps[i + 1]) for i in range(t.n_val)])
    for name, interp, fc in [("Persistence", pers, t.train[-1]), ("OT midpoint", mid, None)]:
        p = f"{ROOT}/generated/{name}/{label}/seed_0.npz"; os.makedirs(os.path.dirname(p), exist_ok=True)
        out = {"interp": interp[:, :, :t.d]}
        if fc is not None: out["forecast"] = fc[:, :t.d]
        np.savez_compressed(p, **out)
    print(label, "done", flush=True)
