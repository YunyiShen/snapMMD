import numpy as np
from SBIRR.Schrodinger import gpIPFP, SchrodingerTrain, gpdrift, nndrift
import torch
import torch.nn as nn
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
import time
import csv
#from SBIRR.gradient_field_NN import train_nn_gradient
from models import SDEfromSBIRR
import torchsde

def train_nn_gradient(model, x_train, y_train_gradient, optimizer = None,
                      epochs = 500, lr = 0.01,verbose=False):
    """
    Train the neural network model to learn the gradient field.
    Args:
        model: The neural network model.
        x_train: The input data for training.
        y_train_gradient: The output data for training (the gradient field)
        epochs: The number of epochs to train the model.
        lr: The learning rate.
        verbose: Whether to print the loss at each epoch.
    Returns:
        The trained neural network model.
    """

    loss_function = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict(x_train)
        # Compute Loss
        loss = loss_function(y_pred, y_train_gradient)
        if verbose:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
    return model

class sbirrlamboseendiv(nn.Module):
    def __init__(self, x0,y0, logscale, circulation, 
                 x0_div, y0_div, logscale_div, divergence):
        super(sbirrlamboseendiv, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))

        self.x0_div = nn.Parameter(torch.tensor(x0_div))
        self.y0_div = nn.Parameter(torch.tensor(y0_div))
        self.logscale_div = nn.Parameter(torch.tensor(logscale_div))
        self.divergence = nn.Parameter(torch.tensor(divergence))


    def forward(self, y):
        scale = torch.exp(self.logscale)
        xx = (y[:,0] - self.x0) * torch.exp(-scale)
        yy = (y[:,1] - self.y0) * torch.exp(-scale)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dxdt = self.circulation * (-dthetadt) * torch.sin(theta)
        dydt = self.circulation * dthetadt * torch.cos(theta)

        scale_div = torch.exp(self.logscale_div)
        xx = (y[:,0] - self.x0_div) * torch.exp(-scale_div)
        yy = (y[:,1] - self.y0_div) 
        dxdt += self.divergence * xx
        dydt += self.divergence * yy

        return torch.stack([dxdt, dydt], dim = 1)
    
    def predict(self, x):
        return self.forward(x)


class sbirrnninput(nn.Module):
    def __init__(self, m_vec, l_vec, sigma_vec, net, smallleakage = 1e-8):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        super(sbirrnninput, self).__init__()

        ## parametric part 
        self.m_vec = nn.Parameter(torch.log(m_vec)) # maximum expression level
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = torch.log(sigma_vec)
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression
        ## net part 
        self.net = net

    def forward(self, x):
        y = x[:, :self.n_gene] # removing time
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        #breakpoint()
        degradation  = torch.relu(y) * l_vec # degredation 
        regulation = torch.sigmoid(self.net(torch.relu(y)))
        production = regulation * m_vec # production

        return production - degradation + self.smallleakage
    
    def predict(self, x):
        return self.forward(x)



def main():
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # determine which task to run
    task_name = sys.argv[3] # GoM and pbmc
    task_name_alt = task_name

    # readin data
    if "pbmc" in task_name:
        data = np.load(f"./data/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"./data/{task_name}_data.npz")
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
        
        sigma = 0.1
        N = 40
        if task_name == 'GoM':
            ours = sbirrlamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0.)
        elif task_name == 'pbmc':
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
            ours = sbirrnninput(m_vec_guess, l_vec_guess, 
                                sigma_vec_guess, mlp)
            N = 25
        else:
            raise NotImplementedError
        oursnndrift = nndrift(ours.double().to(device), 
                              train_nn_gradient, N = N)
        # get problem specifications
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
        np.savez(f"./interpolation/SBIRR_{task_name}_interpolation_{seed}.npz",
                 interpolation = interpolation.cpu().detach().numpy())

        torch.save(SBIRR_model, f"./models/SBIRR_{task_name_alt}_model_{seed}.pt")
        
        
        # IPFP forward 
        sigma = 0.1
        N = 40
        if task_name == 'GoM':
            ours = sbirrlamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0.)
        elif task_name == 'pbmc':
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
            ours = sbirrnninput(m_vec_guess, l_vec_guess, 
                                sigma_vec_guess, mlp)
            N = 25
        else:
            raise NotImplementedError
        oursnndrift = nndrift(ours.double().to(device), 
                              train_nn_gradient, N = N)
        
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