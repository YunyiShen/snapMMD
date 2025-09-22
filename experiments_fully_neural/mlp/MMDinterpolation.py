import torch
import numpy as np
import torchsde 
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from snapMMD.booleansde import nninputfun, mlpproduction, NNdrift, MLP

def get_model(kind, task_name, seed):
    dims = {("classic","Repressilator"): 3,
        ("classic", "LV"): 2,
        ("missingobs", "Repressilator"): 6,
        ("realdata", "GoM"): 2,
        ("realdata", "pbmc"): 30
    }
    torch.manual_seed(0)
    dim = dims[(kind, task_name)]
    net = MLP(dim)
    sigma_vec_guess = 3.3*torch.tensor([.01 for _ in range(dim)])
    
    mymodel = NNdrift(net, sigma_vec_guess)
    model = torch.load(f"./models/{kind}_{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    mymodel.load_state_dict(model)#.to(device)
    
    return mymodel

def get_data(kind, task_name, seed = 42):
    if task_name == "pbmc":
        data = np.load(f"../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../data/{kind}/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    
    if kind != "missingobs":
        X_0 = torch.tensor(data["Xs"][0])
    else:
        X_0 = torch.tensor(data["Xs"][0])
        X_0 = torch.concatenate((X_0, torch.tensor(data['y0'])[:X_0.shape[0], X_0.shape[1]:]), dim = 1)
    #X0 = torch.tensor(data["Xs"][0])
    time_scale = data['time_scale']
    fitted_model = get_model(kind, task_name, seed).float()
    
    #breakpoint()

    return fitted_model, X_0, t_start/time_scale, t_end/time_scale 



seeds = [1, 2, 3, 4, 5,  40,41, 42, 43, 44]
tasks = [#("classic", "Repressilator"),
         #("classic", "LV"),
         ("missingobs", "Repressilator")#,
         #("realdata", "GoM"),
         #("realdata", "pbmc")
         
         ]
for kind, task_name in tasks:
    print(f"task {task_name}")
    for seed_use in tqdm(seeds):
        model, X0, t_start, t_end = get_data(kind, task_name, seed_use)

        dts_save = torch.linspace(t_start, t_end, 500)
        Xs = torchsde.sdeint(model, X0.float(), dts_save.float(), method='euler')
        
        Xs = Xs.detach().numpy()
        np.savez(f"./interpolation/{kind}_{task_name}_interpolation_{seed_use}.npz", 
                 interpolation = Xs)

