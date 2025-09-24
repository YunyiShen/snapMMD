import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
from snapMMD.booleansde import nninputfun, mlpproduction, NNdrift, MLP
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def get_settings(kind, task_name):
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
    lr = 0.0025
    epochs = 1000
    mymodel = NNdrift(net, sigma_vec_guess).to(device)
    
    return lr, epochs, mymodel


def run(id):
    
    all_tasks = [("classic","Repressilator"),
        ("classic", "LV"),
        ("missingobs", "Repressilator")#,
        ("realdata", "GoM"),
        ("realdata", "pbmc")
    ]
    
    seedid = id % 10
    taskid = id // 10
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    
    kind, task_name = all_tasks[taskid]
    print(kind, task_name, seeds[seedid])
    
    
    my_seeds = [seeds[seedid]]
    # grab command line arguments 
    #my_task_id = int(sys.argv[1])
    #num_tasks = int(sys.argv[2])

    # determine which task to run
    #task_name = sys.argv[1]
    if task_name == "pbmc":
        data = np.load(f"../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../data/{kind}/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device).float() for i in range(N_steps-1)] # training data
    #breakpoint()
    X_val = torch.tensor(data["Xs"][-1]).float().to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device).float()
    y0 = torch.tensor(data['y0']).to(device).float()
    time_scale = data['time_scale']
    time_scale = torch.tensor(time_scale).to(device).float()
    lr, epochs, mymodel = get_settings(kind, task_name)
    #my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)
        myDLS = snapMMD(mymodel, Xs, dts[:-1].to(device)/time_scale, lr = lr)
        rbf = RBF().to(device)
        myMMD = MMDLoss(kernel = rbf).to(device)

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False)
        if kind != "missingobs":
            X_0 = Xs[0]
        else:
            X_0 = Xs[0]
            X_0 = torch.concatenate((X_0, y0[:X_0.shape[0], X_0.shape[1]:]), dim = 1)
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/time_scale]).to(device).float(), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{kind}_{task_name}_model_{seed}.pt")
        np.savez(f"./forecasts/{kind}_{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())

import fire           

if __name__ == '__main__':
    fire.Fire(run)

