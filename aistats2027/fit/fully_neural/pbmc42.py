import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
from snapMMD.booleansde import nninputfun
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def get_settings(taskname):
    if "pbmc" in taskname:
        torch.manual_seed(42)
        dim = 30
        net = MLP(dim).to(torch.float64)
        sigma_vec_guess = 3.3*torch.tensor([.01 for _ in range(dim)])
        lr = 0.0025
        epochs = 1000
        mymodel = NNdrift(net, sigma_vec_guess).to(device)
    
    
    
    return lr, epochs, mymodel



def run(id, rep = 5, epochs_override = None, lr_override = None, gradclip = None):
    seedid = id % 10
    taskid = id // 10
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    all_tasks = ["pbmc"]
    task_name = all_tasks[taskid]
    print(task_name, seeds[seedid])
    print(epochs_override, lr_override, gradclip)
    
    my_seeds = [seeds[seedid]]
    
    if "pbmc" in task_name:
        data = np.load(f"../../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../../data/realdata/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    y0 = Xs[0].repeat([rep,1]) #torch.tensor(data['y0']).to(device)
    time_scale = data['time_scale']
    time_scale = torch.tensor(time_scale).to(device)
    lr, epochs, mymodel = get_settings(task_name)
    epochs = epochs_override if epochs_override is not None else epochs
    lr = lr_override if lr_override is not None else lr
    #my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)
        myDLS = snapMMD(mymodel, Xs, dts[:-1].to(device)/time_scale, lr = lr)
        rbf = RBF().to(device)
        myMMD = MMDLoss(kernel = rbf).to(device)

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False, gradclip = gradclip)

        X_0 = Xs[0]
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/time_scale]).to(device), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{task_name}42_rep{rep}_epochs{epochs}_lr{lr}_gradclip{gradclip}_model_{seed}.pt")
        np.savez(f"./forecasts/{task_name}42_rep{rep}_epochs{epochs}_lr{lr}_gradclip{gradclip}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())

from fire import Fire

if __name__ == '__main__':
    Fire(run)
