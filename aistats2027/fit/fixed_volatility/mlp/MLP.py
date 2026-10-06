import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class nninputfunfixvol(nn.Module):
    def __init__(self, m_vec, l_vec, sigma_vec, net, smallleakage = 1e-8, zero_init = False):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        super(nninputfunfixvol, self).__init__()

        ## parametric part 
        self.m_vec = nn.Parameter(torch.log(m_vec)) # maximum expression level
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression
        ## net part 
        self.net = net
        
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        degradation  = torch.relu(y) * l_vec # degredation 
        regulation = torch.sigmoid(self.net(torch.relu(y)))
        production = regulation * m_vec # production

        return production - degradation + self.smallleakage

    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.ones_like(y)) * 0.1


def get_settings(taskname):
    if "Repressilator" in taskname:
        lr = 0.0025
        epochs = 3000
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
        mymodel = nninputfunfixvol( m_vec_guess, 
                                  l_vec_guess, 
                                  sigma_vec_guess, 
                                  net = mlp, 
                                  zero_init = False).to(device) 
    
    return lr, epochs, mymodel


def run(id):
    seedid = id % 10
    taskid = id // 10
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    all_tasks = ["Repressilator"]
    task_name = all_tasks[taskid]
    #print(kind, task_name, seeds[seedid])
    
    
    my_seeds = [seeds[seedid]]

    data = np.load(f"../../../../data/classic/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    y0 = Xs[0].repeat((5,1)) 
    time_scale = data['time_scale']
    time_scale = torch.tensor(time_scale).to(device)
    lr, epochs, mymodel = get_settings(task_name)
    #my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        torch.manual_seed(seed)
        myDLS = snapMMD(mymodel, Xs, dts[:-1].to(device)/time_scale, lr = lr)
        rbf = RBF().to(device)
        myMMD = MMDLoss(kernel = rbf).to(device)

        myDLS.train(myMMD, y0.to(device), epochs = epochs, adaptive_bandwidth = False)

        X_0 = Xs[0]
        forecast = torchsde.sdeint(mymodel, X_0.to(device), torch.tensor([0, dts[-1]/time_scale]).to(device), 
                           method='euler')

        torch.save(mymodel.state_dict(), f"./models/{task_name}_model_{seed}.pt")
        np.savez(f"./forecasts/{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
import fire           

if __name__ == '__main__':
    fire.Fire(run)


