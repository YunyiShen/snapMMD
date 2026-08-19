from dls.dls import MMDLoss, DLS, RBF
import numpy as np
import torch
from TrajectoryNet.optimal_transport.emd import earth_mover_distance
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
        rbf = RBF(bandwidth = 1.)
        myMMD = MMDLoss(kernel = rbf)
        forecast = np.load(f"./{kind}/forecasts/{task_name}_forecast_{seed}.npz")['forecast'][-1]
        #breakpoint()
        if kind == "missingobs":
            #breakpoint()
            forecast = forecast[:, :forecast.shape[-1]//2]
        #breakpoint()
        return myMMD(torch.tensor(X_val), torch.clamp(torch.tensor(forecast), -1e8, 1e8)).cpu().numpy().item()
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
        

