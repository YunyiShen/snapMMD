from snapMMD.dls import MMDLoss, RBF
import numpy as np
import torch
import os


def get_metric(kind, task_name, seed = 42):
    
    if task_name == "pbmc":
        data = np.load(f"../data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
    else:
        if kind == "mlp":
            data = np.load(f"../data/classic/{task_name}_data_interp_val.npz")
        else:
                
            data = np.load(f"../data/{kind}/{task_name}_data_interp_val.npz")
    X_val = data["Xs"] # interpolationing target
    if os.path.exists(f"./interpolation/{kind}_{task_name}_interpolation_{seed}.npz"):
        rbf = RBF(bandwidth = 1.)
        myMMD = MMDLoss(kernel = rbf)
        # full trajectory, shape (n_steps, n_particles, dim)
        interpolation = np.load(f"./interpolation/{kind}_{task_name}_interpolation_{seed}.npz")['interpolation']
        if kind == "missingobs":
            interpolation = interpolation[..., :interpolation.shape[-1]//2]
        # validation snapshot i is compared with the trajectory at the middle of the
        # i-th of n_val equal time segments, the rule used for the main experiments
        n_val = X_val.shape[0]
        points = np.linspace(0, interpolation.shape[0] - 1, n_val + 1, dtype=int)
        idx = (points[1:] + points[:-1]) // 2
        metric = np.array([myMMD(torch.tensor(X_val[i]), 
                      torch.tensor(interpolation[idx[i]])).cpu().numpy().item()
                for i in range(n_val)
                ])
        # no clamping: flag the snapshots of a diverged trajectory instead
        diverged = np.array([not np.isfinite(interpolation[j]).all() or np.abs(interpolation[j]).max() > 1e8
                for j in idx
                ])
        return metric, diverged
    return None

seeds = [1, 2, 3, 4, 5, 40, 42, 43, 44, 41]

all_tasks = [
        ("classic", "LV"),
        ("classic","Repressilator"),
        ("mlp", "Repressilator"),
        ("missingobs", "Repressilator"),
        ("realdata", "GoM"),
        ("realdata", "pbmc")
    ]

label_map = {                               # readable names for summary rows
    ("LV", "classic"):               "Lotka-Volterra",
    ("Repressilator", "classic"):    "Repress. Param.",
    ("Repressilator", "missingobs"): "Repress. Incomplete",
    ("Repressilator", "mlp"):        "Repress. Semiparam.",
    ("GoM", "realdata"):             "Gulf of Mexico",
    ("pbmc", "realdata"):            "PBMC"
}


results = {}
for kind, task in all_tasks:
    metrics = []
    diverged = []
    used_seeds = []
    print(kind, task)
    label = label_map[(task, kind)]
    for seed in seeds:
        metric = get_metric(kind, task, seed)
        if metric is not None:
           metrics.append(metric[0])
           diverged.append(metric[1])
           used_seeds.append(seed)
    metrics = np.array(metrics) # (n_seeds, n_val)
    diverged = np.array(diverged)
    print(f"{len(used_seeds)} seeds, {diverged.sum()} of {diverged.size} snapshots non-finite or beyond 1e8")
    print(f"${metrics.mean():.3f}\\pm{{\\scriptsize {metrics.std():.3f}}}$ & [{metrics.min():.3f}, {metrics.max():.3f}]")    
    results[f"{kind}_{task}"] = metrics
    results[f"{kind}_{task}_seeds"] = np.array(used_seeds)
    results[f"{kind}_{task}_diverged"] = diverged

# per-seed, per-validation-time values
np.savez("./metric_interpol.npz", **results)
        

