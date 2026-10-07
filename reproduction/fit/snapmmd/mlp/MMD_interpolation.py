import torch
import numpy as np
import torchsde 
import matplotlib.pyplot as plt
from tqdm import tqdm
from dls.booleansde import nninputfun
import torch.nn as nn


def get_data(task_name, seed = 42):
    data = np.load(f"./data/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    time_scale = data['time_scale']
    X0 = torch.tensor(data["Xs"][0])
    model = torch.load(f"./models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    if task_name == "Repressilator":
        torch.manual_seed(0)
        m_vec_guess = 10*torch.tensor([5.,5.,5.])
        l_vec_guess = 10*torch.tensor([1., 1., 1.])
        sigma_vec_guess = 3.3*torch.tensor([.01, .01, .01])
        n_gene = 3
        mlp = nn.Sequential(
            nn.Linear(n_gene, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, n_gene)
        )
        fitted_model = nninputfun( m_vec_guess, 
                                  l_vec_guess, 
                                  sigma_vec_guess, 
                                  net = mlp, 
                                  zero_init = False)
        fitted_model.load_state_dict(model)
    else:
        raise ValueError("Unknown task name")

    return fitted_model, X0, t_start/time_scale, t_end/time_scale


seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
task_names = ["Repressilator"]
for task_name in task_names:
    print(f"task {task_name}")
    for seed_use in tqdm(seeds):
        model, X0, t_start, t_end = get_data(task_name, seed_use)

        dts_save = torch.linspace(t_start, t_end, 500)
        Xs = torchsde.sdeint(model, X0, dts_save, method='euler')
        
        Xs = Xs.detach().numpy()
        np.savez(f"./interpolation/{task_name}_interpolation_{seed_use}.npz", 
                 interpolation = Xs)
             
