import torch
import numpy as np
import torchsde 
import matplotlib.pyplot as plt
#from models import LotkaVolterra, repressilator
from tqdm import tqdm
from models import *


def get_data(task_name, seed = 42):
    data = np.load(f"./data/{task_name}_data.npz")
    N_steps = data['N_steps']
    time_scale = data['time_scale']
    Xs =[data["Xs"][i] for i in range(N_steps-1)] # training data
    X_val = data["Xs"][-1] # forecasting target
    forecast = np.load(f"./forecasts/{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbirr = np.load(f"./forecasts/SBIRR_{task_name}_forecast_{seed}.npz")['forecast']
    forecastsbforward = np.load(f"./forecasts/SBforward_{task_name}_forecast_{seed}.npz")['forecast']


    model = torch.load(f"./models/{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    sbirr = torch.load(f"./models/SBIRR_{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))
    sbforward = torch.load(f"./models/SBforward_{task_name}_model_{seed}.pt", map_location=torch.device('cpu'))

    return Xs, X_val, forecast, forecastsbirr, forecastsbforward,\
           model, sbirr, sbforward, time_scale


seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]

for seed_use in tqdm(seeds):
    task_name = "Repressilator"
    Xs, X_val, forecast, forecastsbirr, forecastsbforward,\
    model, sbirr, sbforward, time_scale = get_data(task_name, seed = seed_use)

    repressilator_gt = repressilator( alpha = 1e-5, 
                                 beta_m = 10.,
                                 n = 3., 
                                 k = 1., 
                                 gamma_m = 1., 
                                 beta_p = 1., 
                                 gamma_p = 1., 
                                 sigma = 0.02)
    
    fitted_model = repressilator( alpha = 1e-5, 
                                 beta_m = 10.,
                                 n = 3., 
                                 k = 1., 
                                 gamma_m = 1., 
                                 beta_p = 1., 
                                 gamma_p = 1., 
                                 sigma = 0.02)
    fitted_model.load_state_dict(model)




    range_pred = np.arange(0, 9, 1)
    rna1, rna2,rna3, protein1, protein2, protein3 = np.meshgrid(range_pred, range_pred, 
                             range_pred, range_pred,
                             range_pred, range_pred)
    rna1 = torch.tensor(rna1.flatten())
    rna2 = torch.tensor(rna2.flatten())
    rna3 = torch.tensor(rna3.flatten())
    protein1 = torch.tensor(protein1.flatten())
    protein2 = torch.tensor(protein2.flatten())
    protein3 = torch.tensor(protein3.flatten())

    gt_vector = repressilator_gt.f(0.,torch.stack([rna1, rna2, rna3, protein1, protein2, protein3], dim = 1)).detach().numpy()
    model_vector = fitted_model.f(0.,torch.stack([rna1, rna2, rna3, protein1, protein2, protein3], dim = 1)).detach().numpy()/time_scale
    sbirr_vector = sbirr.f(0.,torch.stack([rna1, rna2, rna3], dim = 1)).detach().numpy()
    sbforward_vector = sbforward.f(0.,torch.stack([rna1, rna2, rna3], dim = 1)).detach().numpy()
    #breakpoint()
    np.savez(f"./vector_fields/{task_name}_vector_field_{seed_use}.npz",
                rna1 = rna1,
                rna2 = rna2,
                rna3 = rna3,
                protein1 = protein1,
                protein2 = protein2,
                protein3 = protein3,
                gt_vector = gt_vector,
                model_vector = model_vector,
                sbirr_vector = sbirr_vector,
                sbforward_vector = sbforward_vector)