import torch
import numpy as np
import torchsde 
import matplotlib.pyplot as plt
from models import LotkaVolterra, repressilator
from tqdm import tqdm


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
    task_name = "LV"
    Xs, X_val, forecast, forecastsbirr, forecastsbforward,\
           model, sbirr, sbforward, time_scale = get_data(task_name, seed_use)

    ground_truth_model = LotkaVolterra(1.0, 0.4, 0.4, 0.1, 0.02)
    fitted_model = LotkaVolterra(.5, .1, .1, .02, .01)
    fitted_model.load_state_dict(model)

    x = np.arange(0, 11, 0.4)
    y = np.arange(1, 5, 0.4)
    xx, yy = np.meshgrid(x, y)
    xx = torch.tensor(xx.flatten())
    yy = torch.tensor(yy.flatten())

    gt_vector = ground_truth_model.f(0.,torch.stack([xx, yy], dim = 1)).detach().numpy()
    model_vector = fitted_model.f(0.,torch.stack([xx, yy], dim = 1)).detach().numpy()/time_scale
    sbirr_vector = sbirr.f(0.,torch.stack([xx, yy], dim = 1)).detach().numpy()
    sbforward_vector = sbforward.f(0.,torch.stack([xx, yy], dim = 1)).detach().numpy()
    np.savez(f"./vector_fields/{task_name}_vector_field_{seed_use}.npz", 
             xx = xx, 
             yy = yy,
             gt_vector = gt_vector, 
             model_vector = model_vector, 
             sbirr_vector = sbirr_vector, 
             sbforward_vector = sbforward_vector)
    
    task_name = "Repressilator"
    Xs, X_val, forecast, forecastsbirr, forecastsbforward,\
           model, sbirr, sbforward, time_scale = get_data(task_name, seed_use)

    range_pred = np.arange(0, 6, 1)
    xx, yy, zz = np.meshgrid(range_pred, range_pred, range_pred)
    xx = torch.tensor(xx.flatten())
    yy = torch.tensor(yy.flatten())
    zz = torch.tensor(zz.flatten())

    ground_truth_model = repressilator(10.,3.,1.,1., 0.02)
    fitted_model = repressilator(10.,1.,1.,10., .03)
    fitted_model.load_state_dict(model)

    gt_vector = ground_truth_model.f(0.,torch.stack([xx, yy, zz], dim = 1)).detach().numpy()
    model_vector = fitted_model.f(0.,torch.stack([xx, yy, zz], dim = 1)).detach().numpy()/time_scale
    sbirr_vector = sbirr.f(0.,torch.stack([xx, yy, zz], dim = 1)).detach().numpy()
    sbforward_vector = sbforward.f(0.,torch.stack([xx, yy, zz], dim = 1)).detach().numpy()

    np.savez(f"./vector_fields/{task_name}_vector_field_{seed_use}.npz", 
             xx = xx, 
             yy = yy, 
             zz = zz,
             gt_vector = gt_vector, 
             model_vector = model_vector, 
             sbirr_vector = sbirr_vector, 
             sbforward_vector = sbforward_vector)
             


