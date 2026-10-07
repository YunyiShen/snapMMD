"""Shared definitions: tasks, data, the evaluation metric (squared MMD with the snapshot bandwidth, App. D.11) and EMD."""
import os
import numpy as np
import ot
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))     # aistats2027/
DATA = os.path.abspath(os.path.join(ROOT, "..", "data"))                  # the released snapshots of the snapMMD repo
SEEDS = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
# paper label -> data file stem (without .npz) under data/
TASKS = {"LV": "classic/LV_data", "ReprParam": "classic/Repressilator_data", "ReprSemiparam": "classic/Repressilator_data",
         "ReprProtein": "missingobs/Repressilator_data", "GoM": "realdata/GoM_data",
         "PBMC": "realdata/processed_pbmc_data_sub500_every_2_until20"}


def median_bandwidth(P, max_points=2000, seed=0):
    """Median squared distance between distinct points of P (fixed random subsample if P has more than max_points)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) > max_points:
        P = P[np.random.default_rng(seed).choice(len(P), max_points, replace=False)]
    D = torch.cdist(torch.tensor(P), torch.tensor(P)) ** 2
    iu = torch.triu_indices(len(P), len(P), offset=1)
    return torch.median(D[iu[0], iu[1]]).item()


class Task:
    """Training snapshots, forecast target, validation snapshots and the training-kernel bandwidth of one task."""

    def __init__(self, label):
        self.label = label
        train = np.load(f"{DATA}/{TASKS[label]}.npz"); val = np.load(f"{DATA}/{TASKS[label]}_interp_val.npz")
        self.train = train["Xs"][:-1]; self.forecast_truth = train["Xs"][-1]; self.val_truth = val["Xs"]
        self.d = train["Xs"].shape[-1]; self.n_val = self.val_truth.shape[0]
        self.h_train = median_bandwidth(self.train.reshape(-1, self.d))   # median heuristic on the pooled training snapshots


def kernel_means(A, B, hs, chunk=2000):
    tot = np.zeros(len(hs))
    for i in range(0, len(A), chunk):
        D = torch.cdist(A[i:i + chunk], B) ** 2
        for j, h in enumerate(hs):
            tot[j] += torch.exp(-D / h).sum().item()
    return tot / (len(A) * len(B))


def mmd2(X, Y, h_train=None):
    """Squared MMD (V-statistic) with a single RBF kernel exp(-|x-y|^2/h). Returns a dict over bandwidths:
    'snap' = median squared distance within the true snapshot X (the paper's metric), 'snap/4', '4snap', and 'h_train'."""
    X = torch.tensor(np.asarray(X, dtype=np.float64)); Y = torch.tensor(np.asarray(Y, dtype=np.float64))
    h = median_bandwidth(X.numpy()); hs = [h, h / 4, 4 * h] + ([h_train] if h_train else [])
    v = kernel_means(X, X, hs) - 2 * kernel_means(X, Y, hs) + kernel_means(Y, Y, hs)
    out = {"snap": v[0], "snap/4": v[1], "4snap": v[2]}
    if h_train: out["h_train"] = v[3]
    return out


def emd(p, q):
    """Earth mover's distance (2-Wasserstein): square root of the exact OT cost with squared Euclidean ground cost."""
    p = np.asarray(p, np.float64); q = np.asarray(q, np.float64)
    M = np.maximum((p * p).sum(1)[:, None] + (q * q).sum(1)[None, :] - 2 * p @ q.T, 0)
    return float(np.sqrt(ot.emd2(np.ones(len(p)) / len(p), np.ones(len(q)) / len(q), np.ascontiguousarray(M), numItermax=int(1e7))))


def midpoints(n_steps, n_val):
    """Index rule for reading a simulated path at the validation times: the middle of each of n_val equal segments."""
    pts = np.linspace(0, n_steps - 1, n_val + 1, dtype=int)
    return (pts[1:] + pts[:-1]) // 2


def save_output(path, d, forecast=None, traj=None, n_val=None):
    """Uniform output format: 'forecast' (N, d) final cloud, 'interp' (n_val, N, d) path at the validation times."""
    out = {}
    if forecast is not None:
        out["forecast"] = np.asarray(forecast)[:, :d]
    if traj is not None:
        out["interp"] = np.asarray(traj)[midpoints(traj.shape[0], n_val)][:, :, :d]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)


FIG_SEED_FORECAST, FIG_SEED_INTERP = 42, 44      # seeds shown in the forecast and interpolation figures


def path_for_figure(traj, d, n_particles=200, n_times=100, dims=3, seed=0):
    """(T, N, D) simulated path -> (n_particles, n_times, min(d, dims)) float32 subsample, used only to draw the trajectory
    figures: particles drawn without replacement (fixed seed), times evenly thinned, the first observed coordinates kept."""
    traj = np.asarray(traj)
    T, N = traj.shape[:2]
    ti = np.unique(np.linspace(0, T - 1, min(T, n_times)).round().astype(int))
    pi = np.sort(np.random.default_rng(seed).choice(N, min(N, n_particles), replace=False))
    return np.ascontiguousarray(traj[ti][:, pi, :min(d, dims)].transpose(1, 0, 2)).astype(np.float32)
