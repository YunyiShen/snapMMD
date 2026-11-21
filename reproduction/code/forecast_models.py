"""Frozen-checkpoint adapters; physical time -> each checkpoint's trained clock."""
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torchsde

from common import ROOT
from simulate import CKPT, MODELS, MLP, NEURAL, NNdrift

SB_KIND = {"LV": "classic", "ReprParam": "classic", "ReprSemiparam": "mlp",
           "ReprProtein": "missingobs", "GoM": "realdata", "PBMC": "realdata"}
_loaded_kind = None


def sb_classes(kind, package, stubs=None):
    """Compatibility loader for trusted released object pickles, as in baseline.py.

    No fitting is called. The external SBIRR package directory is explicit.
    Optional stubs are only needed for unused plotting imports in some installations.
    """
    global _loaded_kind
    if package:
        sys.path.insert(0, str(Path(package).resolve()))
    if stubs:
        sys.path.insert(0, str(Path(stubs).resolve()))
    if _loaded_kind == kind:
        return
    folder = Path(ROOT) / "fit/baselines/schrodinger_bridge" / kind
    sys.modules.pop("models", None)
    sys.path.insert(0, str(folder))
    try:
        importlib.import_module("models")
        spec = importlib.util.spec_from_file_location("forecast_sb_" + kind, folder / "baseline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, val in vars(mod).items():
            if isinstance(val, type) and val.__module__ == mod.__name__:
                setattr(sys.modules["__main__"], name, val)
        _loaded_kind = kind
    finally:
        sys.path.remove(str(folder))


class Adapter:
    def __init__(self, method, task, seed, data, sb_package=None, sb_stubs=None):
        self.method, self.task, self.seed = method, task, seed
        self.observed_dim = data["Xs"].shape[-1]
        self.physical_factor = 20. if task == "PBMC" else 1.
        self.latent = 0
        if method in ("SBIRR-ref", "SB-forward"):
            prefix = "SBIRR" if method == "SBIRR-ref" else "SBforward"
            self.checkpoint = Path(ROOT) / "checkpoints/schrodinger_bridge" / prefix / task / ("model_%d.pt" % seed)
            sb_classes(SB_KIND[task], sb_package, sb_stubs)
            self.model = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.clock_divisor = self.physical_factor  # SB fits the stored data times.
            self.dtype = torch.float64
        else:
            self.checkpoint = Path(ROOT) / "checkpoints" / CKPT[method] / task / ("model_%d.pt" % seed)
            if method == "Fully neural":
                dim, self.latent, self.dtype = NEURAL[task]
                self.model = NNdrift(MLP(dim), 3.3 * torch.tensor([.01] * dim)).to(self.dtype)
            else:
                build, self.latent = MODELS[method][task]
                self.model = build()
                self.dtype = torch.from_numpy(data["Xs"][0]).dtype
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise ValueError("nonfinite checkpoint parameters")
            self.clock_divisor = self.physical_factor * float(data["time_scale"])
        self.model.eval()

    def initial(self, data, protocol):
        if protocol == "last" and self.latent:
            raise ValueError("last restart unavailable: unobserved latent state requires inference")
        index = 0 if protocol == "first" else -2
        x = torch.tensor(data["Xs"][index], dtype=self.dtype)
        if self.latent:
            x = torch.cat((x, torch.zeros(len(x), self.latent, dtype=self.dtype)), 1)
        return x, float(data["dts"][index]) * self.physical_factor

    @torch.no_grad()
    def rollout(self, data, protocol, endpoints, entropy, dt=.001):
        """One uninterrupted rollout, endpoints in physical time, fixed Euler dt.

        Brownian domains always cover first observation through final endpoint;
        equal entropy therefore couples the future increments of first/last starts.
        Times use the original data dtype, matching the released SnapMMD loader.
        """
        x, start = self.initial(data, protocol)
        ts = torch.tensor([start] + list(endpoints), dtype=self.dtype) / self.clock_divisor
        if not torch.all(ts[1:] > ts[:-1]):
            raise ValueError("forecast endpoints must strictly follow the chosen start")
        origin = float(data["dts"][0]) * self.physical_factor / self.clock_divisor
        bm = torchsde.BrownianInterval(t0=origin, t1=float(ts[-1]), size=x.shape,
                    dtype=self.dtype, device=x.device, entropy=int(entropy), halfway_tree=True, tol=1e-6)
        ys = torchsde.sdeint(self.model, x, ts, bm=bm, method="euler", dt=dt, adaptive=False)
        assert torch.equal(ys[0], x), "zero-horizon invariant failed"
        expected = (len(ts), len(x), x.shape[1])
        assert tuple(ys.shape) == expected
        return ys[1:, :, :self.observed_dim].cpu().numpy(), ts.cpu().numpy()
