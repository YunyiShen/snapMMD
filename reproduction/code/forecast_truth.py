"""Independent future synthetic populations from the original physical generator.

Each task/horizon/replicate draws a fresh initial population, matching the unpaired
snapshot construction. Nothing is conditioned on the observed last training cloud.
"""
import numpy as np
import torch
import torchsde
from parameter_recovery import reference_functions

TARGET_IDS = (81001, 81002, 81003)
HORIZONS = (1, 2, 3, 5)


class TruthSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, task):
        super().__init__()
        if task not in ("LV", "ReprParam"):
            raise ValueError(task)
        self.task = task

    def f(self, t, x):
        return reference_functions(self.task, x)[0]

    def g(self, t, x):
        return .02 * torch.relu(x)


def streams(task, horizon, replicate):
    return np.random.SeedSequence([int(replicate), 11 if task == "LV" else 17, int(horizon)]).generate_state(2).tolist()


@torch.no_grad()
def population(task, horizon, replicate, n=200, dt=.001):
    initial_seed, brownian_seed = streams(task, horizon, replicate)
    lower = np.array([5., 4.] if task == "LV" else [1., 1., 2.])
    rng = np.random.default_rng(initial_seed)
    x = torch.tensor(lower + .1 * rng.random((n, len(lower))), dtype=torch.float64)
    endpoint = 9. + horizon
    bm = torchsde.BrownianInterval(t0=0., t1=endpoint, size=x.shape, dtype=x.dtype,
                                 entropy=brownian_seed, halfway_tree=True, tol=1e-6)
    ys = torchsde.sdeint(TruthSDE(task), x, torch.tensor([0., endpoint], dtype=x.dtype),
                       bm=bm, method="euler", dt=dt, adaptive=False)
    assert torch.equal(ys[0], x) and torch.isfinite(ys).all()
    return ys[-1].numpy(), dict(initial_seed=initial_seed, brownian_seed=brownian_seed,
                              physical_endpoint=endpoint, dt=dt, n=n)
