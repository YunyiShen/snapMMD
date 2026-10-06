import numpy as np
from SBIRR.Schrodinger import gpIPFP, SchrodingerTrain, gpdrift, nndrift
import torch
import torch.nn as nn
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
import time
import csv
from SBIRR.experiments_helper import get_settings
from models import SDEfromSBIRR
import torchsde


def main():
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[3]
    if task_name == "repres":
        task_name_alt = "Repressilator" # different naming sadly
    else:
        task_name_alt = task_name

    # readin data
    data = np.load(f"./data/{task_name_alt}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    
    # all seeds
    my_seeds = seeds[my_task_id:len(seeds):num_tasks]

    for seed in my_seeds:
        print(f"task {task_name_alt} with seed {seed}")
        # set seed
        torch.manual_seed(seed)

        # get problem specifications
        sigma, _, N, _, oursnndrift, _ = get_settings(task_name, N_steps)
        #breakpoint()
        if task_name == "repres":
            N = 40
            oursnndrift.dt = 1./N
        #breakpoint()
        def gpIPFPmaker(ref_drift = None, N = N, device = device):
            return gpIPFP(
                  ref_drift = ref_drift, 
                  N = N, device = device)
        #breakpoint()
        oursSchrodinger_train = SchrodingerTrain(Xs, dts[:-1].detach().cpu().numpy(), sigma)

        # IRR
        start_time = time.time()
        drift, interpolation = oursSchrodinger_train.iter_drift_fit(gpIPFPmaker,
                                                                oursnndrift,
                                                                ipfpiter = 10, 
                                                                iteration = 10)
        end_time = time.time()
        elapsed_time_irr = end_time - start_time

        # dump model
        SBIRR_model = SDEfromSBIRR(drift, sigma)


        X_0 = Xs[0]
        forecast = torchsde.sdeint(SBIRR_model, X_0.to(device), torch.tensor([0, dts[-1]]).to(device), 
                           method='euler')

        np.savez(f"./forecasts/SBIRR_{task_name_alt}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
        np.savez(f"./interpolation/SBIRR_{task_name_alt}_interpolation_{seed}.npz",
                 interpolation = interpolation.cpu().detach().numpy())


        torch.save(SBIRR_model, f"./models/SBIRR_{task_name_alt}_model_{seed}.pt")
        

        # IPFP forward 
        sigma, _, N, _, oursnndrift, _ = get_settings(task_name, N_steps)
        #breakpoint()
        if task_name == "repres":
            N = 50
            oursnndrift.dt = 1./N
        def gpIPFPmaker(ref_drift = None, N = N, device = device):
            return gpIPFP(
                  ref_drift = ref_drift, 
                  N = N, device = device)
        oursSchrodinger_train = SchrodingerTrain(Xs, dts[:-1].detach().cpu().numpy(), sigma)

        start_time = time.time()
        drift = oursSchrodinger_train.IPFP_forward_learning(gpIPFPmaker,
                                                                oursnndrift,
                                                                iteration = 10)
        end_time = time.time()
        elapsed_time_ipfp = end_time - start_time
        SBforward_model = SDEfromSBIRR(drift, sigma)
        X_0 = Xs[0]
        forecast = torchsde.sdeint(SBforward_model, X_0.to(device), torch.tensor([0, dts[-1]]).to(device), 
                           method='euler')

        np.savez(f"./forecasts/SBforward_{task_name_alt}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
        # dump model
        torch.save(SBforward_model, f"./models/SBforward_{task_name_alt}_model_{seed}.pt")

        #  IPFP_forward_learning


if __name__ == "__main__":
    main()