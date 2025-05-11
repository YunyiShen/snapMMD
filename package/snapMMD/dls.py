import torch
import torch.nn as nn
import torchsde
from tqdm import tqdm
import numpy as np

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=1.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            #n_samples = L2_distances.shape[0]
            #return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
            with torch.no_grad():
                return torch.median(L2_distances)
        return self.bandwidth
    
    def get_bandwidth_from_data(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        #n_samples = L2_distances.shape[0]
        #return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        with torch.no_grad():
            return torch.median(L2_distances)

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / ((self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[:, None, None])).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF(bandwidth=1.)): # default to a fixed bandwidth for optimization
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
class snapMMD:
    def __init__(self, sde, marginals, dts, bm = None, method='euler', 
                 optimizer = None, lr = None):
        '''
        sde: an sde object
        marginals: list of marginal samples as torch tensor 
        dts: time when margianls are taken
        bm: brownian motion
        method: method for sde 
        optimizer: an optimizer
        '''
        self.sde = sde
        self.marginals = marginals
        self.n = dts.shape[0]
        self.dts = dts
        self.bm = bm
        self.method = method
        if lr is None:
            lr = 0.01
        if optimizer is None:
            optimizer = torch.optim.Adam(sde.parameters(), lr = lr)
        self.optimizer = optimizer

    def train(self, MMD,y0 = None,epochs = 10, weights = None, obsdims = None, adaptive_bandwidth = False, frozen_step = None, trim = None):
        if y0 is None:
            y0 = self.marginals[0]
        if obsdims is None:
            obsdims = self.marginals[0].shape[1]
        if weights is None:
            weights = torch.tensor( [margin.shape[0] for margin in self.marginals]) # get sample size for each marginal
            weights = weights/weights.sum() # normalize
            weights = torch.pow(weights, 2) # square
            weights = weights/weights.sum() # normalize
            weights = weights.to(y0.device)

        
        if self.bm is None:
            self.bm = torchsde.BrownianInterval(t0=self.dts[0], t1=self.dts[-1], 
                                                size=(y0.shape[0], y0.shape[1]), device = y0.device)
        
        if adaptive_bandwidth:
            if MMD.kernel.bandwidth is not None:
                Warning('Bandwidth is set to a fixed value, and adaptive is turned on, we will start with that fixed value')
                if frozen_step is None:
                    frozen_step = [ int(epochs/4), int(epochs/2), int(3 * epochs/4)] # default to freeze bandwidth at 1/4, 1/2, 3/4 epochs
            else:
                MMD.kernel.bandwidth = MMD.kernel.get_bandwidth_from_data(torch.cat(self.marginals))
                if frozen_step is None:
                    frozen_step = [ int(epochs/4), int(epochs/2), int(3 * epochs/4)]
            if trim is None:
                trim = [0.1, 500.]
        else:
            if MMD.kernel.bandwidth is None:
                MMD.kernel.bandwidth = MMD.kernel.get_bandwidth_from_data(torch.cat(self.marginals))
            
        pooled_marginals = torch.cat(self.marginals)
        ssr = torch.tensor(0).double().to(y0.device)
        for i in range(self.n):
            ssr += weights[i] * MMD(pooled_marginals, self.marginals[i])
        del pooled_marginals
        pbar = tqdm(range(epochs))
        for tt in pbar:
            ys = torchsde.sdeint_adjoint(self.sde, y0, self.dts, 
                                     method=self.method, bm=self.bm)
            loss = 0.0
            frozen_counter = 0
            if adaptive_bandwidth and frozen_counter < len(frozen_step):
                if tt in frozen_step:
                    bandwidth_all = []
                    for i in range(self.n):
                        bandwidth_all += [MMD.kernel.get_bandwidth_from_data(torch.vstack([ys[i][:,:obsdims], self.marginals[i]]))]
                    
                    
                    nominal_bandwidth = torch.median(torch.stack(bandwidth_all))
                    nominal_bandwidth = torch.clamp(nominal_bandwidth, trim[0], trim[1])
                    #breakpoint()

                    MMD.kernel.bandwidth = nominal_bandwidth

                    pooled_marginals = torch.cat(self.marginals)
                    # recompute ssr
                    ssr = torch.tensor(0).double().to(y0.device)
                    for i in range(self.n):
                        ssr += weights[i] * MMD(pooled_marginals, self.marginals[i])
                    del pooled_marginals

                    frozen_counter += 1
                    

            for i in range(self.n):
                loss += weights[i] * MMD(ys[i][:,:obsdims], self.marginals[i])
            if torch.isnan(loss):
                #breakpoint()
                for name, param in self.sde.named_parameters():
                    if param.requires_grad:
                        print([name, param.data])

            rsqr = 1.-loss/ssr
            #print(loss.item(), rsqr.item(), MMD.kernel.bandwidth)
            pbar.set_description(f"rsqr:{rsqr.item()} with bandwidth {MMD.kernel.bandwidth}" )
            self.optimizer.zero_grad()
        # Backward pass
            loss.backward(retain_graph=False)
            self.optimizer.step()
