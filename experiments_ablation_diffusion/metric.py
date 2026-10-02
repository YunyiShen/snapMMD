from snapMMD.dls import evaluation_mmd
import numpy as np
import torch
import os


def get_metric(kind, task_name, seed = 42):
    
    if task_name == "pbmc":
        data = np.load(f"../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        if kind == "mlp":
            data = np.load(f"../data/classic/{task_name}_data.npz")
        else:
                
            data = np.load(f"../data/{kind}/{task_name}_data.npz")
    X_val = data["Xs"][-1] # forecasting target
    if os.path.exists(f"./{kind}/forecasts/{task_name}_forecast_{seed}.npz"):
        forecast = np.load(f"./{kind}/forecasts/{task_name}_forecast_{seed}.npz")['forecast'][-1]
        if kind == "missingobs":
            forecast = forecast[:, :forecast.shape[-1]//2]
        # squared MMD, single RBF kernel, bandwidth by the median heuristic on the true snapshot; no clamping
        return evaluation_mmd(torch.tensor(X_val), torch.tensor(forecast)).cpu().numpy().item()
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


for kind, task in all_tasks:
    metrics = []
    print(kind, task)
    label = label_map[(task, kind)]
    for seed in seeds:
        metric = get_metric(kind, task, seed)
        if metric is not None:
           metrics.append(metric) 
    metrics = np.array(metrics)
    print(f"${metrics.mean():.3f}\\pm{{\\scriptsize {metrics.std():.3f}}}$ & [{metrics.min():.3f}, {metrics.max():.3f}]")    
        

