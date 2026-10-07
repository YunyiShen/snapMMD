import torch
import torch.nn as nn
import torchsde


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