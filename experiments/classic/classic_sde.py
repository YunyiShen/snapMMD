import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
import sys
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_settings(taskname):
    if "LV" in taskname:
        lr = 0.05
        epochs = 300
        mymodel = LotkaVolterra(.5 * 9, .1 * 9, .1 * 9, .02 * 9, .01 * 3).to(device)
    if "Repressilator" in taskname:
        lr = 0.05
        epochs = 500
        mymodel = repressilator(10.,1.,1.,10., .03).to(device) 
    
    return lr, epochs, mymodel


def main():
    seeds = [42] #[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    #my_task_id = int(sys.argv[1])
    #num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[1]

    data = np.load(f"../../data/classic/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    y0 = torch.tensor(data['y0']).to(device)
    time_scale = data['time_scale']
    lr, epochs, mymodel = get_settings(task_name)
    my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)
        #breakpoint()
        myDLS = snapMMD(mymodel, Xs, dts[:-1].to(device)/torch.tensor(time_scale).to(device), lr = lr)
        rbf = RBF().to(device)
        myMMD = MMDLoss(kernel = rbf).to(device)

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False)

        X_0 = Xs[0]
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/torch.tensor(time_scale).to(device)]).to(device), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{task_name}_model_{seed}.pt")
        np.savez(f"./forecasts/{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())

if __name__ == '__main__':
    main()
