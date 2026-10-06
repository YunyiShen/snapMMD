import torch.nn as nn
import torch
import torchsde 

class LotkaVolterra(nn.Module):

    def __init__(self, alpha, beta, gamma, delta, sigma):
        super(LotkaVolterra, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.delta = nn.Parameter(torch.log(torch.tensor(delta)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        alpha = self.preprocess(self.alpha) # things needs to be positive
        beta = self.preprocess(self.beta)
        gamma = self.preprocess(self.gamma)
        delta = self.preprocess(self.delta)
        y = torch.relu(y) + 1e-10 # avoid going to 0
        dxdt =  alpha * y[:,0] - beta * y[:,0] * y[:,1]
        dydt = delta * y[:,0] * y[:,1] - gamma * y[:,1]
        return torch.stack([dxdt, dydt], dim = 1)
    
    def g(self, t, y):
        sigma = self.preprocess(self.sigma)
        return sigma * torch.relu(y)
    

class lorenz(nn.Module):
    def __init__(self, alpha, beta, rho, sigma):
        super(lorenz, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.rho = nn.Parameter(torch.log(torch.tensor(rho)))
        
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        
        beta = self.preprocess(self.beta)
        alpha = self.preprocess(self.alpha)
        rho = self.preprocess(self.rho)
        
        #y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt =  alpha * (y[:,1] - y[:,0])
        dydt = y[:,0] * (rho - y[:,2]) - y[:,1]
        dzdt = y[:,0] * y[:,1] - beta * y[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        return  sigma + 0 * torch.relu(y)
    

class repressilator(nn.Module):
    def __init__(self, beta, n, k, gamma, sigma):
        super(repressilator, self).__init__()
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.n = nn.Parameter(torch.log(torch.tensor(n)))
        self.k = nn.Parameter(torch.log(torch.tensor(k)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        
        beta = self.preprocess(self.beta)
        n = self.preprocess(self.n)
        k = self.preprocess(self.k)
        gamma = self.preprocess(self.gamma)
        y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt = beta/(1.+ (y[:,2]/k) ** n) - gamma * y[:,0]
        dydt = beta/(1.+ (y[:,0]/k) ** n) - gamma * y[:,1]
        dzdt = beta/(1.+ (y[:,1]/k) ** n) - gamma * y[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        return sigma * torch.relu(y)

class SDEfromSBIRR(nn.Module):
    def __init__(self, drift, sigma):
        super(SDEfromSBIRR, self).__init__()
        self.drift = drift
        self.sigma = sigma
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        #breakpoint()
        #x = torch.concat([y, t[:,None]], dim = 1)
        return self.drift(y)
    
    def g(self, t, y):
        return self.sigma * torch.ones_like(y) 