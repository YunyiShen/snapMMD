import torchsde
import torch.nn as nn
import torch
import numpy as np
from snapMMD.dls import MMDLoss, snapMMD, RBF
from snapMMD.booleansde import nninputfun
import sys
from models import lamboseen, helmholtz, lamboseendiv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_settings(taskname):
    if "GoM" in taskname:
        lr = 0.05
        epochs = 300
        #mymodel = lamboseen(0., 0., -1.5, -1., .03).to(device) 
        #mymodel = helmholtz(0., 0., -1.5, -1.5, 0., 0., .01).to(device)
        mymodel = lamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0., .01).to(device)
    
    if "pbmc" in taskname:
        lr = 0.001
        epochs = 6000
        torch.manual_seed(0)
        n_gene = 30
        m_vec_guess = 20*torch.ones(n_gene) * 5.
        l_vec_guess = 20*torch.ones(n_gene)
        sigma_vec_guess = np.sqrt(20)*torch.ones(n_gene) * .01
        mlp = nn.Sequential(
            nn.Linear(n_gene, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, n_gene)
        ).to(torch.float64)
        mymodel = nninputfun( m_vec_guess, 
                                  l_vec_guess, 
                                  sigma_vec_guess, 
                                  net = mlp, 
                                  zero_init = False).to(device) 

    return lr, epochs, mymodel



def main():
    seeds = [42] #[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    #my_task_id = int(sys.argv[1])
    #num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[1]
    if "pbmc" in task_name:
        data = np.load(f"../../data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"../../data/realdata/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).to(device) for i in range(N_steps-1)] # training data
    
    X_val = torch.tensor(data["Xs"][-1]).to(device) # forecasting target

    dts = torch.tensor(data['dts']).to(device)
    y0 = torch.tensor(data['y0']).to(device)
    time_scale = data['time_scale']
    time_scale = torch.tensor(time_scale).to(device)
    lr, epochs, mymodel = get_settings(task_name)
    my_seeds = seeds#[my_task_id:len(seeds):num_tasks]
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

if __name__ == '__main__':
    main()
