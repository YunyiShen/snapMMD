import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
from snapMMD.booleansde import nninputfun, mlpproduction
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MLP(nn.Module):
    def __init__(self, dim, mid=[64]):
        super().__init__()
        layers = []
        prev_dim = dim
        for m in mid:
            layers.append(nn.Linear(prev_dim, m))
            layers.append(nn.ReLU())
            prev_dim = m
        layers.append(nn.Linear(prev_dim, dim))  # final layer back to dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NNdrift(nn.Module):
    def __init__(self, net, sigma_vec):
        super().__init__()
        self.net = net
        self.preprocesspos = torch.exp
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    
    def f(self, t, y):
        return self.net(y)
    
    def g(self, t, y):
        
        sigma = self.preprocesspos(self.sigma_vec)
        return torch.ones_like(y) * sigma

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


def run(id, gradclip = None):
    
    all_tasks = [("classic","Repressilator"),
        ("missingobs", "Repressilator"),
        ("classic", "LV"),
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
        data = np.load(f"../../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../../data/{kind}/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device).float() for i in range(N_steps-1)] # training data
    #breakpoint()
    X_val = torch.tensor(data["Xs"][-1]).float().to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device).float()
    y0 = Xs[0].repeat((5,1))  #torch.tensor(data['y0']).to(device).float()
    if kind == "missingobs":
        y0 = torch.concatenate([y0, torch.zeros_like(y0)], axis = 1)
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

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False, gradclip = gradclip)
        if kind != "missingobs":
            X_0 = Xs[0]
        else:
            X_0 = Xs[0]
            X_0 = torch.concatenate((X_0, y0[:X_0.shape[0], X_0.shape[1]:]), dim = 1)
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/time_scale]).to(device).float(), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{kind}_{task_name}_gradclip{gradclip}_model_{seed}.pt")
        np.savez(f"./forecasts/{kind}_{task_name}_gradclip{gradclip}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())

import fire           

if __name__ == '__main__':
    fire.Fire(run)

