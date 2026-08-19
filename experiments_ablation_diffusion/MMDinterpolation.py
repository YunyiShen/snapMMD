import torch
import numpy as np
import torchsde 
import matplotlib.pyplot as plt
from classic.models import repressilator as repsclassic
from classic.models import LotkaVolterra
from missingobs.models import repressilator as repsmis
from snapMMD.booleansde import nninputfun
from realdata.models import lamboseendiv
from tqdm import tqdm
from torch import nn


def get_data_classic(task_name, seed = 42):
    data = np.load(f"../data/classic/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    X0 = torch.tensor(data["Xs"][0])
    time_scale = data['time_scale']
    model = torch.load(f"./classic/models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    if task_name == "LV":
        fitted_model = LotkaVolterra(.5, .1, .1, .02, .01)
        fitted_model.load_state_dict(model)
    elif task_name == "Repressilator":
        fitted_model = repsclassic(10.,1.,1.,10., .03)
        fitted_model.load_state_dict(model)
    else:
        raise ValueError("Unknown task name")
    #breakpoint()

    return fitted_model, X0, t_start/time_scale, t_end/time_scale

def get_data_mlp(task_name, seed = 42):
    data = np.load(f"../data/classic/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    time_scale = data['time_scale']
    X0 = torch.tensor(data["Xs"][0])
    model = torch.load(f"./mlp/models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
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


def get_data_missingobs(task_name, seed = 42):
    data = np.load(f"../data/missingobs/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    time_scale = data['time_scale']
    X0 = torch.tensor(data['y0'])
    model = torch.load(f"./missingobs/models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    
    if task_name == "Repressilator":
        fitted_model = repsmis( alpha = 1e-5, 
                                 beta_m = 10.,
                                 n = 3., 
                                 k = 1., 
                                 gamma_m = 1., 
                                 beta_p = 1., 
                                 gamma_p = 1., 
                                 sigma = 0.02)
        fitted_model.load_state_dict(model)
    else:
        raise ValueError("Unknown task name")

    return fitted_model, X0, t_start/time_scale, t_end/time_scale


def get_data_realdata(task_name, seed = 42):
    if "pbmc" in task_name:
        data = np.load(f"../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../data/realdata/{task_name}_data.npz")
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    time_scale = data['time_scale']
    X0 = torch.tensor(data["Xs"][0])
    model = torch.load(f"./realdata/models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    if task_name == "pbmc":
        torch.manual_seed(0)
        n_gene = 30
        m_vec_guess = 20*torch.ones(n_gene) * 5.
        l_vec_guess = 20*torch.ones(n_gene)
        sigma_vec_guess = np.sqrt(20)*torch.ones(n_gene) * .01
        mlp = nn.Sequential(
            nn.Linear(n_gene, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, n_gene)
        ).to(torch.float64)
        fitted_model = nninputfun( m_vec_guess, 
                                  l_vec_guess, 
                                  sigma_vec_guess, 
                                  net = mlp, 
                                  zero_init = False)
        fitted_model.load_state_dict(model)
    elif task_name == "GoM":
        fitted_model = lamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0., .01)
        fitted_model.load_state_dict(model)
    else:
        raise ValueError("Unknown task name")

    return fitted_model, X0, t_start/time_scale, t_end/time_scale


def get_data(kind, task_name, seed = 42):
    if kind == "classic":
        return get_data_classic(task_name, seed)
    if kind == "missingobs":
        return get_data_missingobs(task_name, seed)

    if kind == "mlp":
        return get_data_mlp(task_name, seed)
    
    if kind == "realdata":
        return get_data_realdata(task_name, seed)
    
    raise ValueError("Unknown task name")

    



seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
tasks = [#("classic", "Repressilator"),
         #("classic", "LV"),
         #("missingobs", "Repressilator"),
         ("mlp", "Repressilator"), 
         ("realdata", "GoM"),
         ("realdata", "pbmc")
         
         ]
for kind, task_name in tasks:
    print(f"task {task_name}")
    for seed_use in tqdm(seeds):
        model, X0, t_start, t_end = get_data(kind, task_name, seed_use)

        dts_save = torch.linspace(t_start, t_end, 500)
        Xs = torchsde.sdeint(model, X0, dts_save, method='euler')
        
        Xs = Xs.detach().numpy()
        np.savez(f"./interpolation/{kind}_{task_name}_interpolation_{seed_use}.npz", 
                 interpolation = Xs)

