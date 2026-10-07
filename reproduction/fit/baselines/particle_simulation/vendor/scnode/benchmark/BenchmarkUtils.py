# scNODE (Zhang & Singh 2024), MIT licence, github.com/rsinghlab/scNODE. Only sampleGaussian is kept from benchmark/BenchmarkUtils.py (the original imports scanpy).
import torch
import torch.distributions as dist

def sampleGaussian(mean, std):
    '''
    Sampling with the re-parametric trick.
    '''
    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1)
    x = r * std.float() + mean.float()
    return x
