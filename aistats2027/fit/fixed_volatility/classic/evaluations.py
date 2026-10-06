from dls.dls import MMDLoss, DLS, RBF
import numpy as np
import torch
from TrajectoryNet.optimal_transport.emd import earth_mover_distance


def get_data(task_name, seed = 42):
    data = np.load(f"./data/{task_name}_data.npz")
    X_val = data["Xs"][-1] # forecasting target
    forecast = np.load(f"./forecasts/{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbirr = np.load(f"./forecasts/SBIRR_{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbforward = np.load(f"./forecasts/SBforward_{task_name}_forecast_{seed}.npz")['forecast']
    vector_fileds = np.load(f"./vector_fields/{task_name}_vector_field_{seed}.npz")


    return X_val, forecast, forecastsbirr, forecastsbforward,\
            vector_fileds


seeds = [1, 2, 3, 4, 5, 40, 42, 43, 44, 41]
tasks = ["LV", "Repressilator"]

mmd_l2 = {"LV": [], "Repressilator":[]}
mmd_mmd = {"LV": [], "Repressilator":[]}
mmd_emd = {"LV": [], "Repressilator":[]}

sbirr_l2 = {"LV": [], "Repressilator":[]}
sbirr_mmd = {"LV": [], "Repressilator":[]}
sbirr_emd = {"LV": [], "Repressilator":[]}


sbforward_l2 = {"LV": [], "Repressilator":[]}
sbforward_mmd = {"LV": [], "Repressilator":[]}
sbforward_emd = {"LV": [], "Repressilator":[]}

rbf = RBF(bandwidth = 1.)
myMMD = MMDLoss(kernel = rbf)

for task in tasks:
    for seed in seeds:
        X_val, forecast, forecastsbirr,\
              forecastsbforward, vector_fields = get_data(task, seed)
        gt_vector = vector_fields['gt_vector']
        model_vector = vector_fields['model_vector']
        sbirr_vector = vector_fields['sbirr_vector']
        sbforward_vector = vector_fields['sbforward_vector']

        mmd_l2[task].append(np.mean((gt_vector - model_vector)**2))
        sbirr_l2[task].append( np.mean((gt_vector - sbirr_vector)**2))
        sbforward_l2[task].append( np.mean((gt_vector - sbforward_vector)**2))
        #breakpoint()
        mmd_mmd[task].append(myMMD(torch.tensor(X_val), 
                                   torch.tensor(forecast[-1])).detach().numpy().item())
        sbirr_mmd[task].append(myMMD(torch.tensor(X_val), 
                                   torch.tensor(forecastsbirr[-1])).detach().numpy().item())
        sbforward_mmd[task].append(myMMD(torch.tensor(X_val), 
                                   torch.tensor(forecastsbforward[-1])).detach().numpy().item())
        mmd_emd[task].append(earth_mover_distance(X_val, forecast[-1]))
        sbirr_emd[task].append(earth_mover_distance(X_val, forecastsbirr[-1]))
        sbforward_emd[task].append(earth_mover_distance(X_val, forecastsbforward[-1]))

    # Compute comprehensive statistics for L2 metrics
    mmd_l2_vals = np.array(mmd_l2[task])
    mmd_l2[task] = {
        'individual': mmd_l2[task].copy(),
        'mean': np.mean(mmd_l2_vals),
        'std': np.std(mmd_l2_vals),
        'min': np.min(mmd_l2_vals),
        'max': np.max(mmd_l2_vals),
        'range': np.max(mmd_l2_vals) - np.min(mmd_l2_vals)
    }
    
    sbirr_l2_vals = np.array(sbirr_l2[task])
    sbirr_l2[task] = {
        'individual': sbirr_l2[task].copy(),
        'mean': np.mean(sbirr_l2_vals),
        'std': np.std(sbirr_l2_vals),
        'min': np.min(sbirr_l2_vals),
        'max': np.max(sbirr_l2_vals),
        'range': np.max(sbirr_l2_vals) - np.min(sbirr_l2_vals)
    }
    
    sbforward_l2_vals = np.array(sbforward_l2[task])
    sbforward_l2[task] = {
        'individual': sbforward_l2[task].copy(),
        'mean': np.mean(sbforward_l2_vals),
        'std': np.std(sbforward_l2_vals),
        'min': np.min(sbforward_l2_vals),
        'max': np.max(sbforward_l2_vals),
        'range': np.max(sbforward_l2_vals) - np.min(sbforward_l2_vals)
    }
    
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
    
    sbirr_mmd_vals = np.array(sbirr_mmd[task])
    sbirr_mmd[task] = {
        'individual': sbirr_mmd[task].copy(),
        'mean': np.mean(sbirr_mmd_vals),
        'std': np.std(sbirr_mmd_vals),
        'min': np.min(sbirr_mmd_vals),
        'max': np.max(sbirr_mmd_vals),
        'range': np.max(sbirr_mmd_vals) - np.min(sbirr_mmd_vals)
    }
    
    sbforward_mmd_vals = np.array(sbforward_mmd[task])
    sbforward_mmd[task] = {
        'individual': sbforward_mmd[task].copy(),
        'mean': np.mean(sbforward_mmd_vals),
        'std': np.std(sbforward_mmd_vals),
        'min': np.min(sbforward_mmd_vals),
        'max': np.max(sbforward_mmd_vals),
        'range': np.max(sbforward_mmd_vals) - np.min(sbforward_mmd_vals)
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
    
    sbirr_emd_vals = np.array(sbirr_emd[task])
    sbirr_emd[task] = {
        'individual': sbirr_emd[task].copy(),
        'mean': np.mean(sbirr_emd_vals),
        'std': np.std(sbirr_emd_vals),
        'min': np.min(sbirr_emd_vals),
        'max': np.max(sbirr_emd_vals),
        'range': np.max(sbirr_emd_vals) - np.min(sbirr_emd_vals)
    }
    
    sbforward_emd_vals = np.array(sbforward_emd[task])
    sbforward_emd[task] = {
        'individual': sbforward_emd[task].copy(),
        'mean': np.mean(sbforward_emd_vals),
        'std': np.std(sbforward_emd_vals),
        'min': np.min(sbforward_emd_vals),
        'max': np.max(sbforward_emd_vals),
        'range': np.max(sbforward_emd_vals) - np.min(sbforward_emd_vals)
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

print_comprehensive_stats("MMD-L2", mmd_l2)
print_comprehensive_stats("MMD-MMD", mmd_mmd)  
print_comprehensive_stats("MMD-EMD", mmd_emd)
print_comprehensive_stats("SBIRR-L2", sbirr_l2)
print_comprehensive_stats("SBIRR-MMD", sbirr_mmd)
print_comprehensive_stats("SBIRR-EMD", sbirr_emd)
print_comprehensive_stats("SBforward-L2", sbforward_l2)
print_comprehensive_stats("SBforward-MMD", sbforward_mmd)
print_comprehensive_stats("SBforward-EMD", sbforward_emd)


    

    


