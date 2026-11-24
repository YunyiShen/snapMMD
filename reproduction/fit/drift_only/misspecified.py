"""Restricted drift families; coefficients are represented in solver time."""
import math

import numpy as np
import torch
from torch import nn


def training_saturation_scale(xs):
    """Median prey abundance from already split training snapshots only."""
    if xs.ndim != 3 or xs.shape[-1] != 2:
        raise ValueError("Saturated LV requires [time, particle, 2] training states")
    prey = xs[..., 0].detach().cpu().numpy()
    value = float(np.median(prey))
    if not np.isfinite(prey).all() or not math.isfinite(value) or value <= 0:
        raise ValueError("Training prey median must be finite and positive")
    return value


class SaturatingLV(nn.Module):
    def __init__(self, saturation_scale, time_scale=9.):
        super().__init__()
        if not math.isfinite(saturation_scale) or saturation_scale <= 0:
            raise ValueError("Saturation scale must be finite and positive")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Time scale must be finite and positive")
        self.register_buffer("saturation_scale", torch.tensor(saturation_scale))
        # Same physical coefficient guesses as the original LV fitter.
        for name, value in [("alpha", .5), ("beta", .1), ("gamma", .1), ("delta", .02)]:
            setattr(self, name, nn.Parameter(torch.log(torch.tensor(value * time_scale))))

    def forward(self, y):
        y = torch.relu(y) + 1e-10
        x, predator = y[:, 0], y[:, 1]
        interaction = x * predator / (1. + x / self.saturation_scale)
        return torch.stack([self.alpha.exp() * x - self.beta.exp() * interaction,
                            self.delta.exp() * interaction - self.gamma.exp() * predator], dim=1)


class ExponentialRepressilator(nn.Module):
    def __init__(self, time_scale=10.):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Time scale must be finite and positive")
        # Original Repr fitter guesses: beta=10, k=1, gamma=10 in solver time
        # when the released time_scale is 10. k is a concentration, not a rate.
        self.beta = nn.Parameter(torch.log(torch.tensor(1. * time_scale)))
        self.k = nn.Parameter(torch.log(torch.tensor(1.)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(1. * time_scale)))

    def forward(self, y):
        y = torch.relu(y) + 1e-8
        inhibitor = y[:, [2, 0, 1]]
        return self.beta.exp() * torch.exp(-inhibitor / self.k.exp()) - self.gamma.exp() * y
