from dls.dls import MMDLoss, DLS, RBF
import numpy as np
import torch
from TrajectoryNet.optimal_transport.emd import earth_mover_distance



def get_data(kind, task_name, seed = 42):
    if task_name == "pbmc":
        data = np.load(f"../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../data/{kind}/{task_name}_data.npz")
    X_val = data["Xs"][-1] # forecasting target
    forecast = np.load(f"./forecasts/{kind}_{task_name}_forecast_{seed}.npz")['forecast']
    
    
    #breakpoint()
    return X_val, forecast

seeds = [1, 2, 3, 4, 5, 40, 42, 43, 44, 41]

all_tasks = [("classic","Repressilator"),
        ("classic", "LV"),
        ("missingobs", "Repressilator")#,
        ("realdata", "GoM"),
        ("realdata", "pbmc")
    ]


