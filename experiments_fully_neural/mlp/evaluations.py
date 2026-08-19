from dls.dls import MMDLoss, DLS, RBF
import numpy as np
import torch
from TrajectoryNet.optimal_transport.emd import earth_mover_distance
import os

def get_data(task_name, seed = 42):
    # print the current working directory
    print(os.getcwd())
    data = np.load(f"../../Supercloud/mlp/data/{task_name}_data.npz")
    X_val = data["Xs"][-1] # forecasting target
    forecast = np.load(f"forecasts/{task_name}_forecast_{seed}.npz")['forecast']
    return X_val, forecast


seeds = [1, 2, 3, 4, 5, 40, 41, 42]#, 43, 44, 41]
tasks = ["Repressilator"]

mmd_mmd = { "Repressilator":[]}
mmd_emd = { "Repressilator":[]}

rbf = RBF(bandwidth = 1.)
myMMD = MMDLoss(kernel = rbf)

for task in tasks:
    for seed in seeds:
        X_val, forecast = get_data(task, seed)

        mmd_mmd[task].append(myMMD(torch.tensor(X_val), 
                                   torch.tensor(forecast[-1])).detach().numpy().item())

        mmd_emd[task].append(earth_mover_distance(X_val, forecast[-1]))


    
    
    

    
    # Compute comprehensive statistics for MMD metrics
    mmd_mmd_vals = np.array(mmd_mmd[task])
    mmd_mmd[task] = {
        'individual': mmd_mmd[task].copy(),
        'mean': np.mean(mmd_mmd_vals),
        'std': np.std(mmd_mmd_vals),
        'min': np.min(mmd_mmd_vals),
        'max': np.max(mmd_mmd_vals),
        'range': np.max(mmd_mmd_vals) - np.min(mmd_mmd_vals)
    }

    
    # Compute comprehensive statistics for EMD metrics
    mmd_emd_vals = np.array(mmd_emd[task])
    mmd_emd[task] = {
        'individual': mmd_emd[task].copy(),
        'mean': np.mean(mmd_emd_vals),
        'std': np.std(mmd_emd_vals),
        'min': np.min(mmd_emd_vals),
        'max': np.max(mmd_emd_vals),
        'range': np.max(mmd_emd_vals) - np.min(mmd_emd_vals)
    }
    

# Function to print comprehensive statistics
def print_comprehensive_stats(metric_name, metric_dict):
    print(f"\n=== {metric_name} ===")
    for task, stats in metric_dict.items():
        if isinstance(stats, dict):
            print(f"\n{task}:")
            print(f"  Individual values: {stats['individual']}")
            print(f"  Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}] (span: {stats['range']:.6f})")
        else:
            print(f"{task}: {stats}")

print_comprehensive_stats("MMD-MMD", mmd_mmd)  
print_comprehensive_stats("MMD-EMD", mmd_emd)

    


