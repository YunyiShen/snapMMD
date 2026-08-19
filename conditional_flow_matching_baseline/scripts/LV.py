#!/usr/bin/env python
# Gulf of Mexico Series Interpolation Script
# Based on the GoM.ipynb notebook
# This script runs OT-CFM, SB-CFM and SF2M on the Gulf of Mexico data
# Takes a seed as input and saves models and trajectories without outputting figures

import os
import argparse
import numpy as np
import torch
import torchsde
from tqdm import tqdm
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from torchcfm.utils import torch_wrapper

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size=(3, 32, 32), sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    # Drift
    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x).flatten(start_dim=1) + self.score(x).flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(y) * self.sigma

def parse_args():
    parser = argparse.ArgumentParser(description='Lotka Volterra Series Interpolation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma value for flow matching')
    parser.add_argument('--dim', type=int, default=2, help='Dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for trajectories')
    parser.add_argument('--num_steps', type=int, default=400, help='Number of steps for trajectories')
    parser.add_argument('--num_iters', type=int, default=10000, help='Number of training iterations')
    return parser.parse_args()


def get_data():
    data = np.load(f"../../Supercloud/classic/data/LV_data.npz")
    data_val = np.load(f"../../Supercloud/classic/data/LV_data_interp_val.npz")
    return data, data_val


def get_batch(FM, X, batch_size, n_times, return_noise=False):
    """Construct a batch with points from each timepoint pair"""
    ts = []
    xts = []
    uts = []
    noises = []
    for t_start in range(n_times - 1):
        x0 = (
            torch.from_numpy(X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)])
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut


def train_ot_cfm(args, X, n_times, device):
    print("Training OT-CFM...")
    # Set up model and optimizer
    ot_cfm_model = MLP(dim=args.dim, time_varying=True, w=args.hidden_dim).to(device)
    ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), 1e-4)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.sigma)
    
    # Training loop
    for i in tqdm(range(args.num_iters)):
        ot_cfm_optimizer.zero_grad()
        t, xt, ut = get_batch(FM, X, args.batch_size, n_times)
        vt = ot_cfm_model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()
    
    # Save model
    torch.save(ot_cfm_model.state_dict(), f"../models/LV/ot_cfm_seed_{args.seed}.pt")
    
    # Generate trajectory
    node_ot_cfm = NeuralODE(torch_wrapper(ot_cfm_model), solver="dopri5", sensitivity="adjoint")
    with torch.no_grad():
        traj_ot_cfm = node_ot_cfm.trajectory(
            torch.from_numpy(X[0][:args.num_samples]).float().to(device),
            t_span=torch.linspace(0, n_times - 1, args.num_steps),
        ).cpu()
    
    # Save trajectory
    np.savez(f"../results/LV/ot_cfm_traj_seed_{args.seed}.npz", trajectory=traj_ot_cfm.numpy())
    
    return ot_cfm_model


def train_sf2m(args, X, n_times, device):
    print("Training SF2M...")
    # Set up model and optimizer
    sf2m_model = MLP(dim=args.dim, time_varying=True, w=args.hidden_dim).to(device)
    sf2m_score_model = MLP(dim=args.dim, time_varying=True, w=args.hidden_dim).to(device)
    sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_model.parameters()) + list(sf2m_score_model.parameters()), 1e-4)
    SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=args.sigma)
    
    # Training loop
    max_norm_ut = torch.tensor(0.0)
    for i in tqdm(range(args.num_iters)):
        sf2m_optimizer.zero_grad()
        t, xt, ut, eps = get_batch(SF2M, X, args.batch_size, n_times, return_noise=True)
        lambda_t = SF2M.compute_lambda(t % 1)
        vt = sf2m_model(torch.cat([xt, t[:, None]], dim=-1))
        st = sf2m_score_model(torch.cat([xt, t[:, None]], dim=-1))
        flow_loss = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        if i % 1000 == 0:
            # print(max_norm_ut)
            print(f"{i}: {flow_loss.item():0.2f}, {score_loss.item():0.2f}")
        loss = flow_loss + score_loss
        loss.backward()
        sf2m_optimizer.step()
    
    # Save model
    torch.save(sf2m_model.state_dict(), f"../models/LV/sf2m_seed_{args.seed}.pt")
    
    # Generate trajectory for SB-CFM
    node_sf2m = NeuralODE(torch_wrapper(sf2m_model), solver="euler", sensitivity="adjoint")
    x0 = torch.from_numpy(X[0][:args.num_samples]).float().to(device)
    with torch.no_grad():
        traj_sb_cfm = node_sf2m.trajectory(
            x0,
            t_span=torch.linspace(0, n_times - 1, args.num_steps, device=device),
        ).cpu()
    
    # Save trajectory
    np.savez(f"../results/LV/sb-cfm_traj_seed_{args.seed}.npz", trajectory=traj_sb_cfm.numpy())

    # Generate trajectory for SF2M
    sde = SDE(sf2m_model, sf2m_score_model, input_size=(args.dim,), sigma=args.sigma)
    with torch.no_grad():
        sde_traj = torchsde.sdeint(
            sde,
            x0.to(device),
            ts=torch.linspace(0, n_times - 1, args.num_steps, device=device),
        ).cpu()
    
    # Save trajectory
    np.savez(f"../results/LV/sf2m_traj_seed_{args.seed}.npz", trajectory=sde_traj.numpy())

    return sf2m_model


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs("../models/LV", exist_ok=True)
    os.makedirs("../results/LV", exist_ok=True)
    
    # Check for CUDA
    global device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data, data_val = get_data()
    dts = torch.tensor(data['dts'])
    t_start = dts[0]
    t_end = dts[-2]
    X0 = torch.tensor(data["Xs"][0])
    time_scale = data['time_scale']
    
    # Process data
    n_times = len(dts)
    X = [data['Xs'][t] for t in range(n_times)]
    
    # Train models and generate trajectories
    ot_cfm_model = train_ot_cfm(args, X, n_times, device)
    sf2m_model = train_sf2m(args, X, n_times, device)
    
    print(f"Models and trajectories for LV saved with seed {args.seed}")


if __name__ == "__main__":
    main()

