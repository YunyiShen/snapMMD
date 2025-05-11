import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
import sys
from models import *

def get_settings(taskname):
    if "Repressilator" in taskname:
        lr = 0.05 #0.05
        epochs = 500
        mymodel = repressilator( alpha = 1e-5, 
                                 beta_m = 10., # scale by time scale
                                 n = 1., 
                                 k = 1., 
                                 gamma_m = 10., 
                                 beta_p = 10., 
                                 gamma_p = 10., 
                                 sigma = 0.09).to(device) 
    
    return lr, epochs, mymodel

def main():
    seeds = [42] #[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    #my_task_id = int(sys.argv[1])
    #num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[1]

    data = np.load(f"../../data/missingobs/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target
    #breakpoint()
    dts = torch.tensor(data['dts']).to(device)
    y0 = torch.tensor(data['y0']).to(device)
    time_scale = data['time_scale']
    #breakpoint()
    time_scale = torch.tensor(time_scale).to(device)
    lr, epochs, mymodel = get_settings(task_name)
    my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)
        #breakpoint()
        myDLS = snapMMD(mymodel, Xs, dts[:-1].to(device)/time_scale, lr = lr)
        rbf = RBF().to(device)
        myMMD = MMDLoss(kernel = rbf).to(device)

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False)

        X_0 = Xs[0]
        X_0 = torch.concatenate((X_0, y0[:X_0.shape[0], X_0.shape[1]:]), dim = 1)
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/time_scale]).to(device), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{task_name}_model_{seed}.pt")
        np.savez(f"./forecasts/{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())

if __name__ == '__main__':
    main()
