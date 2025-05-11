import torch
import torch.nn as nn
import torchsde
from tqdm import tqdm

### mrna only boolean SDE
class mrnabooleansde(nn.Module):
    def __init__(self, m_vec, alpha_vec, l_vec, int_mat, stim_mat,n_vec, 
                 k_vec, sigma_vec, smallvalforlogit = 0.00001):
        """
        a mRNA only boolean SDE

        args:
            m_vec: scaling factor of self production
            alpha_vec: binary, whether the rna having leakage 
            l_vec: rate rna decay
            int_mat: interaction matrix, should be binary and having diagonal being 0
                int_mat[i,j] is whether mrna j regulates mrna i (in some way)
            stim_mat: stimulating matrix, should also be binary and having diagonal being 0
                stim_mat[i,j] is whether mrna j stimulate mrna i, effectively int_mat will mask stim_mat
            n_vec: hill coefficients for each mrna
            k_vec: hill threshold for each mrna
            sigma_vec: noise level, vector or scalar
        """
        self.n_gene = m_vec.shape[0]
        assert alpha_vec.shape[0] == self.n_gene 
        assert l_vec.shape[0] == self.n_gene 
        assert int_mat.shape[0] == self.n_gene
        assert int_mat.shape[1] == self.n_gene
        assert stim_mat.shape[0] == self.n_gene
        assert stim_mat.shape[1] == self.n_gene
        assert n_vec.shape[0] == self.n_gene
        assert k_vec.shape[0] == self.n_gene
        assert sigma_vec.shape[0] == self.n_gene

        super(mrnabooleansde, self).__init__()

        int_mat = int_mat * (1 - 2*smallvalforlogit) + smallvalforlogit # this will make 0 to smallvalforlogt and 1 to 1-smallvalforlogit
        self.int_mat = nn.Parameter(torch.logit(int_mat)) # the "binary" interaction matrix is not soften
        
        stim_mat = stim_mat * (1 - 2*smallvalforlogit) + smallvalforlogit # this will make 0 to smallvalforlogt and 1 to 1-smallvalforlogit
        self.stim_mat = nn.Parameter(torch.logit(stim_mat)) # the "binary" interaction matrix is not soften
        alpha_vec = alpha_vec * (1 - 2*smallvalforlogit) + smallvalforlogit
        self.alpha_vec = nn.Parameter(torch.logit(alpha_vec))

        self.m_vec = nn.Parameter(torch.log(m_vec))
        
        self.l_vec = nn.Parameter(torch.log(l_vec))
        self.n_vec = nn.Parameter(torch.log(n_vec))
        self.k_vec = nn.Parameter(torch.log(k_vec))
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))

        self.preprocesspos = torch.exp
        self.preprocessbin = torch.sigmoid
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        ## preprocessing binary things
        int_mat = self.preprocessbin(self.int_mat)
        stim_mat = self.preprocessbin(self.stim_mat) * int_mat # stimulation matrix is masked by the graph
        alpha_vec = self.preprocessbin(self.alpha_vec)

        
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        degradation  = torch.relu(y) * l_vec
        ## get hill equation 
        n_vec = self.preprocesspos(self.n_vec)
        y = torch.relu(y) + 1e-8 # just to make sure things don't get too small
        hill_y = torch.exp( (torch.log(y) - self.k_vec) * n_vec) # (y/k)^n, the hillized response

        ## coefficients for production degredation
        
        
        
        ## actual drift
        regulation = (torch.matmul(hill_y, stim_mat) +  alpha_vec)/(torch.matmul(hill_y, int_mat) + 1.) # boolean regulation
        production = regulation * m_vec # production
        
        if torch.tensor([torch.isnan(production).any(), torch.isnan(degradation).any()]).any():
            #breakpoint()
            print(production)
            print(degradation)
        return production - degradation + 1e-10 # add some leakage
    
    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma
    
    def interaction_mats(self):
        return self.preprocessbin(self.int_mat), self.preprocessbin(self.stim_mat)
    

### semi parametric activation
class mlpinputfun2(nn.Module):
    def __init__(self, m_vec, l_vec, sigma_vec, hidden = 100,smallleakage = 1e-8, zero_init = False):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        super(mlpinputfun2, self).__init__()

        # parametric part
        self.m_vec = nn.Parameter(torch.log(m_vec)) # maximum expression level
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression

        ## activation function
        self.fc1 = nn.Linear(self.n_gene, hidden)
        self.fchalf = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, self.n_gene)

        if zero_init:
            self.fc1.weight.data.fill_(0.)
            self.fchalf.weight.data.fill_(0.)
            self.fc2.weight.data.fill_(0.)
            self.fc1.bias.data.fill_(0.)
            self.fchalf.bias.data.fill_(0.)
            self.fc2.bias.data.fill_(0.)

        self.activation = torch.relu
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        degradation  = torch.relu(y) * l_vec # degredation 
        regulation = torch.sigmoid(self.fc2( self.activation( self.fchalf( self.activation(self.fc1(torch.relu(y)))))))
        production = regulation * m_vec # production

        return production - degradation + self.smallleakage

    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma

class nninputfun(nn.Module):
    def __init__(self, m_vec, l_vec, sigma_vec, net, smallleakage = 1e-8, zero_init = False):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        super(nninputfun, self).__init__()

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
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma

class mlpproduction(nn.Module):
    def __init__(self, l_vec, sigma_vec, hidden = 100,smallleakage = 1e-8, zero_init = False):
        self.n_gene = l_vec.shape[0]
        assert sigma_vec.shape[0] == self.n_gene
        super(mlpproduction, self).__init__()

        # parametric part 
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression

        ## activation function
        self.fc1 = nn.Linear(self.n_gene, hidden)
        self.fchalf = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, self.n_gene)

        if zero_init:
            self.fc1.weight.data.fill_(0.)
            self.fc2.weight.data.fill_(0.)
            self.fc1.bias.data.fill_(0.)
            self.fc2.bias.data.fill_(0.)

        self.activation = torch.relu
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        l_vec = self.preprocesspos(self.l_vec)
        degradation  = torch.relu(y) * l_vec # degredation 
        production = torch.relu(self.fc2( self.activation( self.fchalf( self.activation(self.fc1(torch.relu(y)))))))
        

        return production - degradation + self.smallleakage

    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma


    
class contIsing(nn.Module):
    def __init__(self, m_vec, alpha_vec, l_vec, 
                 int_mat, sigma_vec,smallleakage = 1e-8):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        assert int_mat.shape[0] == self.n_gene
        assert int_mat.shape[1] == self.n_gene
        assert alpha_vec.shape[0] == self.n_gene
        super(contIsing, self).__init__()

        
        self.m_vec = nn.Parameter(torch.log(m_vec)) # maximum expression level
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.alpha_vec = nn.Parameter(alpha_vec)
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression

        ## activation function
        self.int_mat = nn.Parameter( int_mat)

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        degradation  = torch.relu(y) * l_vec # degredation 
        regulation = torch.sigmoid(torch.matmul(torch.relu(y), self.int_mat) + self.alpha_vec)
        production = regulation * m_vec # production

        return production - degradation + self.smallleakage

    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma
    
class contIsingHill(nn.Module):
    def __init__(self, m_vec, alpha_vec, l_vec, n_vec, 
                 k_vec,
                 int_mat, sigma_vec,smallleakage = 1e-8):
        self.n_gene = m_vec.shape[0]
        assert l_vec.shape[0] == self.n_gene 
        assert sigma_vec.shape[0] == self.n_gene
        assert int_mat.shape[0] == self.n_gene
        assert int_mat.shape[1] == self.n_gene
        assert alpha_vec.shape[0] == self.n_gene
        assert n_vec.shape[0] == self.n_gene
        assert k_vec.shape[0] == self.n_gene
        super(contIsingHill, self).__init__()

        
        self.m_vec = nn.Parameter(torch.log(m_vec)) # maximum expression level
        self.l_vec = nn.Parameter(torch.log(l_vec)) # degradation
        self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.alpha_vec = nn.Parameter(alpha_vec)
        self.preprocesspos = torch.exp
        self.smallleakage = smallleakage # some small leakage expression

        ## activation function
        self.n_vec = nn.Parameter((n_vec))
        self.k_vec = nn.Parameter(torch.log(k_vec))
        self.int_mat = nn.Parameter( int_mat)

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        m_vec = self.preprocesspos(self.m_vec)
        l_vec = self.preprocesspos(self.l_vec)
        n_vec = self.preprocesspos(self.n_vec)
        degradation  = torch.relu(y) * l_vec # degredation 
        y = torch.relu(y) + 1e-8 # just to make sure things don't get too small
        hill_y = torch.exp( (torch.log(y) - self.k_vec) * torch.abs(n_vec)) # (y/k)^n, the hillized response

        regulation = torch.sigmoid(torch.matmul(hill_y, self.int_mat) + self.alpha_vec)
        production = regulation * m_vec # production

        return production - degradation + self.smallleakage

    def g(self, t,y):
        sigma = self.preprocesspos(self.sigma_vec)
        return (torch.sqrt(torch.relu(y)+1e-8)) * sigma