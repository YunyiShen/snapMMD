import torch
import torch.nn as nn
import torchsde


class linear(nn.Module):

    def __init__(self, dim):
        super(linear, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.fc.weight.data.fill_(0.0)
        self.logsigma = nn.Parameter(torch.rand(dim))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        return self.fc(y)
    
    def g(self, t, y):
        return torch.exp(self.logsigma) * torch.ones_like(y)


class lamboseen(nn.Module):
    def __init__(self, x0,y0, logscale, circulation, logsigma):
        super(lamboseen, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))
        self.logsigma = nn.Parameter(torch.tensor(logsigma))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        scale = torch.exp(self.logscale)
        xx = (y[:,0] - self.x0) * torch.exp(-scale)
        yy = (y[:,1] - self.y0) * torch.exp(-scale)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dxdt = -dthetadt * torch.sin(theta)
        dydt = dthetadt * torch.cos(theta)
        return self.circulation * torch.stack([dxdt, dydt], dim = 1)
    def g(self, t, y):
        return torch.exp(self.logsigma) * torch.ones_like(y)
    

class lamboseendiv(nn.Module):
    def __init__(self, x0,y0, logscale, circulation, 
                 x0_div, y0_div, logscale_div, divergence,
                 logsigma):
        super(lamboseendiv, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))

        self.x0_div = nn.Parameter(torch.tensor(x0_div))
        self.y0_div = nn.Parameter(torch.tensor(y0_div))
        self.logscale_div = nn.Parameter(torch.tensor(logscale_div))
        self.divergence = nn.Parameter(torch.tensor(divergence))

        self.logsigma = nn.Parameter(torch.tensor(logsigma))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
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
    def g(self, t, y):
        return torch.exp(self.logsigma) * torch.ones_like(y)
    
class helmholtz(nn.Module):
    def __init__(self, x0,y0, logscale_curl, logscale_div, 
                 circulation, divergence, 
                 logsigma):
        super(helmholtz, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale_curl = nn.Parameter(torch.tensor(logscale_curl))
        self.logscale_div = nn.Parameter(torch.tensor(logscale_div))
        self.circulation = nn.Parameter(torch.tensor(circulation))
        self.divergence = nn.Parameter(torch.tensor(divergence))
        self.logsigma = nn.Parameter(torch.tensor(logsigma))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        scale_curl = torch.exp(self.logscale_curl)
        xx = (y[:,0] - self.x0) 
        yy = (y[:,1] - self.y0) * torch.exp(-scale_curl)
        # constant curl field
        dxdt = self.circulation * yy
        dydt = -self.circulation * xx

        scale_div = torch.exp(self.logscale_div)
        xx = (y[:,0] - self.x0) 
        yy = (y[:,1] - self.y0) * torch.exp(-scale_div)
        # constant divergence field
        dxdt += self.divergence * xx
        dydt += self.divergence * yy
        return torch.stack([dxdt, dydt], dim = 1)
    
    def g(self, t, y):
        return torch.exp(self.logsigma) * torch.ones_like(y)
    

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

