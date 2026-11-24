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


def training_lv_followup_scales(xs):
    """Fixed scales for calibrated LV, using only the supplied training split."""
    saturation = training_saturation_scale(xs)
    prey = xs[..., 0].detach().cpu().numpy()
    reference = float(np.median(prey[0]))
    capacity = 5. * float(np.max(prey))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Initial prey median must be finite and positive")
    if not math.isfinite(capacity) or capacity <= reference:
        raise ValueError("Training-derived capacity must exceed the initial prey median")
    return saturation, reference, capacity


class SaturatingLV(nn.Module):
    def __init__(self, saturation_scale, time_scale=9., *, reference_prey=None, carrying_capacity=None):
        super().__init__()
        if not math.isfinite(saturation_scale) or saturation_scale <= 0:
            raise ValueError("Saturation scale must be finite and positive")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Time scale must be finite and positive")
        if reference_prey is not None and (not math.isfinite(reference_prey) or reference_prey <= 0):
            raise ValueError("Reference prey must be finite and positive")
        if carrying_capacity is not None and (reference_prey is None or
                not math.isfinite(carrying_capacity) or carrying_capacity <= reference_prey):
            raise ValueError("Logistic capacity must exceed a positive reference prey")
        self.register_buffer("saturation_scale", torch.tensor(saturation_scale))
        self.register_buffer("reference_prey", None if reference_prey is None else torch.tensor(reference_prey))
        self.register_buffer("carrying_capacity", None if carrying_capacity is None else torch.tensor(carrying_capacity))
        # Defaults preserve the original guesses and arithmetic. New modes match
        # the bilinear initial field at reference_prey, without using true rates.
        multiplier = 1. if reference_prey is None else 1. + reference_prey / saturation_scale
        alpha = .5 if carrying_capacity is None else .5 / (1. - reference_prey / carrying_capacity)
        for name, value in [("alpha", alpha), ("beta", .1 * multiplier),
                            ("gamma", .1), ("delta", .02 * multiplier)]:
            setattr(self, name, nn.Parameter(torch.log(torch.tensor(value * time_scale))))

    def forward(self, y):
        y = torch.relu(y) + 1e-10
        x, predator = y[:, 0], y[:, 1]
        if self.reference_prey is None:
            interaction = x * predator / (1. + x / self.saturation_scale)
        else:
            # Divide before multiplying: finite x*y need not be representable
            # even when the saturated interaction is. No state clipping.
            interaction = (x / (1. + x / self.saturation_scale)) * predator
        growth = self.alpha.exp() * x
        if self.carrying_capacity is not None:
            growth = growth * (1. - x / self.carrying_capacity)
        return torch.stack([growth - self.beta.exp() * interaction,
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
