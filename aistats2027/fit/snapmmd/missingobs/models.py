import torch.nn as nn
import torch
import torchsde 

class langevinlamboseen(nn.Module):

    def __init__(self, x0, y0, logscale, circulation, logsigma, logdrag, logtimescale = 0.):
        super(langevinlamboseen, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))
        self.logsigma = nn.Parameter(torch.tensor(logsigma))
        self.logdrag = nn.Parameter(torch.tensor(logdrag))
        self.logtimescale = nn.Parameter(torch.tensor(logtimescale))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        self.timescale = torch.exp(self.logtimescale)
        scale = torch.exp(self.logscale)
        xx = (y[:,0] - self.x0) * torch.exp(-scale)
        yy = (y[:,1] - self.y0) * torch.exp(-scale)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dudt = -self.circulation * dthetadt * torch.sin(theta) - torch.exp(self.logdrag) * y[:,2]
        dvdt = self.circulation * dthetadt * torch.cos(theta) - torch.exp(self.logdrag) * y[:,3]
        dxdt = y[:, 2]
        dydt = y[:, 3]
        return self.timescale * torch.stack([dxdt, dydt, dudt, dvdt], dim = 1)
    
    def g(self, t, y):
        tmp = torch.exp(self.logsigma) * torch.ones_like(y)
        tmp[:, :2] = 1e-8 # small noise for position
        return tmp




class repressilator(nn.Module):

    def __init__(self, alpha, beta_m, n, k, gamma_m, beta_p, gamma_p, sigma):
        super(repressilator, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta_m = nn.Parameter(torch.log(torch.tensor(beta_m)))
        self.beta_p = nn.Parameter(torch.log(torch.tensor(beta_p)))
        self.n = nn.Parameter(torch.log(torch.tensor(n)))
        self.k = nn.Parameter(torch.log(torch.tensor(k)))
        self.gamma_m = nn.Parameter(torch.log(torch.tensor(gamma_m)))
        self.gamma_p = nn.Parameter(torch.log(torch.tensor(gamma_p)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        alpha = self.preprocess(self.alpha)
        beta_m = self.preprocess(self.beta_m)
        beta_p = self.preprocess(self.beta_p)
        n = self.preprocess(self.n)
        k = self.preprocess(self.k)
        gamma_m = self.preprocess(self.gamma_m)
        gamma_p = self.preprocess(self.gamma_p)
        y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt = alpha + beta_m/(1.+ (y[:,2+3]/k) ** n) - gamma_m * y[:,0]
        dydt = alpha + beta_m/(1.+ (y[:,0+3]/k) ** n) - gamma_m * y[:,1]
        dzdt = alpha + beta_m/(1.+ (y[:,1+3]/k) ** n) - gamma_m * y[:,2]

        dxpdt = beta_p * y[:,0]-gamma_p * y[:,0+3]
        dypdt = beta_p * y[:,1]-gamma_p * y[:,1+3]
        dzpdt = beta_p * y[:,2]-gamma_p * y[:,2+3]
        return torch.stack([dxdt, dydt, dzdt, dxpdt, dypdt, dzpdt], dim = 1)
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
        #x = torch.concat([y, t[:,None]], dim = 1)
        return self.drift(y)
    
    def g(self, t, y):
        return self.sigma * torch.ones_like(y) 
