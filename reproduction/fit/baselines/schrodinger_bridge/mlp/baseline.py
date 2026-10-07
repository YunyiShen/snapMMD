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
#from SBIRR.gradient_field_NN import train_nn_gradient

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

class mlpinput(nn.Module):
    def __init__(self, m_vec_guess, l_vec_guess, net, zero_init = False):
        super(mlpinput, self).__init__()
        self.m_vec = nn.Parameter(torch.log(m_vec_guess))
        self.l_vec = nn.Parameter(torch.log(l_vec_guess))
        self.net = net
        if zero_init:
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data.zero_()
    
    def forward(self, x):
        activation = torch.sigmoid(self.net(x[:,:3]))
        return torch.exp(self.m_vec)*activation - torch.exp(self.l_vec) * torch.relu(x[:,:3])
    
    def predict(self, x):
        return self.forward(x)


def main():
    seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
    # grab command line arguments 
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    task_name = "Repressilator"
    data = np.load(f"./data/Repressilator_data.npz")
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
        N = 30
        sigma = 0.1
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

        m_vec_guess = 1*torch.tensor([5.,5.,5.])
        l_vec_guess = 1*torch.tensor([1., 1., 1.])
        ours = mlpinput( m_vec_guess, 
                         l_vec_guess, 
                         net = mlp, 
                         zero_init = False)

        
        
        oursnndrift = nndrift(ours.double().to(device), 
                              train_nn_gradient, N = N)
        
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


        X_0 = Xs[0].double()
        forecast = torchsde.sdeint(SBIRR_model, X_0.to(device), torch.tensor([0, dts[-1]]).to(device), 
                           method='euler')

        np.savez(f"./forecasts/SBIRR_{task_name}_forecast_{seed}.npz", 
                 forecast = forecast.cpu().detach().numpy(), 
                 X_val = X_val.cpu().detach().numpy())
        np.savez(f"./interpolation/SBIRR_{task_name}_interpolation_{seed}.npz",
                 interpolation = interpolation.cpu().detach().numpy())


        torch.save(SBIRR_model, f"./models/SBIRR_{task_name}_model_{seed}.pt")

        ### IPFP forward ###
        mlp = nn.Sequential(
            nn.Linear(n_gene, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, n_gene)
        )

        m_vec_guess = torch.tensor([5.,5.,5.])
        l_vec_guess = torch.tensor([1., 1., 1.])
        ours = mlpinput( m_vec_guess, 
                         l_vec_guess, 
                         net = mlp, 
                         zero_init = False)

        

        oursnndrift = nndrift(ours.double().to(device), 
                              train_nn_gradient, N = N)
        
        def gpIPFPmaker(ref_drift = None, N = N, device = device):
            return gpIPFP(
                  ref_drift = ref_drift, 
                  N = N, device = device)
        #breakpoint()
        oursSchrodinger_train = SchrodingerTrain(Xs, dts[:-1].detach().cpu().numpy(), sigma)

        start_time = time.time()
        drift = oursSchrodinger_train.IPFP_forward_learning(gpIPFPmaker,
                                                                oursnndrift,
                                                                iteration = 10)
        end_time = time.time()
        elapsed_time_ipfp = end_time - start_time
        SBforward_model = SDEfromSBIRR(drift, sigma)
        X_0 = Xs[0].double()
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
