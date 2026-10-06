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
from SBIRR.gradient_field_NN import train_nn_gradient
from models import SDEfromSBIRR
import torchsde


class sbirrlamboseen(nn.Module):
    def __init__(self, x0,y0, logscale, circulation):
        super(sbirrlamboseen, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))


    def forward(self, y):
        scale = torch.exp(self.logscale)
        xx = (y[:,0] - self.x0) * torch.exp(-scale)
        yy = (y[:,1] - self.y0) * torch.exp(-scale)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dxdt = self.circulation * (-dthetadt) * torch.sin(theta)
        dydt = self.circulation * dthetadt * torch.cos(theta)

        return torch.stack([dxdt, dydt], dim = 1)
    
    def predict(self, x):
        return self.forward(x)


def get_settings_local(taskname, N_steps):
    if "Repressilator" in taskname:
        sigma, _, N, _, oursnndrift, _ = get_settings("repres", N_steps)
        #breakpoint()
        N = 40
        oursnndrift.dt = 1./N
    if "vortex" in taskname:
        sigma = 0.1
        N = 50
        ours = sbirrlamboseen(0., 0., -1.5, -1.5)
        oursnndrift = nndrift(ours.double().to(device), 
                              train_nn_gradient, N = N)
        
    return sigma, N, oursnndrift



def main():
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[3]
    
    # readin data
    data = np.load(f"./data/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    
    # all seeds
    my_seeds = seeds[my_task_id:len(seeds):num_tasks]

    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)

        # get problem specifications
        sigma, N, oursnndrift = get_settings_local(task_name, N_steps)
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

        np.savez(f"./forecasts/SBIRR_{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
        np.savez(f"./interpolation/SBIRR_{task_name}_interpolation_{seed}.npz",
                 interpolation = interpolation.cpu().detach().numpy())

        torch.save(SBIRR_model, f"./models/SBIRR_{task_name}_model_{seed}.pt")
        

        # IPFP forward 
        sigma, N, oursnndrift = get_settings_local(task_name, N_steps)
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

        np.savez(f"./forecasts/SBforward_{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
        # dump model
        torch.save(SBforward_model, f"./models/SBforward_{task_name}_model_{seed}.pt")

        #  IPFP_forward_learning


if __name__ == "__main__":
    main()