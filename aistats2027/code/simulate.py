"""Simulate forecasts and interpolation paths (and R^2) from the shipped checkpoints of SnapMMD and its ablations.
Writes generated/{method}/{task}/seed_{seed}.npz in the uniform format of outputs/ and generated/r2.csv.

Protocol (as for every number in the paper): start from the observed first snapshot (unobserved protein coordinates at 0);
forecast = sdeint(model, X0, [0, t_T]) with the Euler scheme; path = sdeint(model, X0, linspace(t_1, t_{T-1}, 500)), read at the
validation times; times divided by the task's time scale. Seeds: seed (forecast), seed + 1000 (path), seed + 2000 (R^2).
For seed 44 it also writes generated/paths_for_figures/{method}/{task}.npz (a subsample of the path, for the trajectory figures).
Usage: python simulate.py [--methods Ours "Fixed volatility" ...] [--tasks LV ...] [--seeds 44 ...]  (R^2 is written only when all seeds run)
"""
import argparse, csv, importlib.util, os, sys
import numpy as np
import torch, torch.nn as nn, torchsde
from snapMMD.booleansde import nninputfun
from snapMMD.dls import MMDLoss, RBF
from common import ROOT, DATA, SEEDS, TASKS, FIG_SEED_INTERP, path_for_figure, save_output

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


classic = load("classic", f"{HERE}/models/classic.py"); missing = load("missing", f"{HERE}/models/missingobs.py")
realdata = load("realdata", f"{HERE}/models/realdata.py")
fv_classic = load("fv_classic", f"{HERE}/models/fixed_volatility/classic.py"); fv_missing = load("fv_missing", f"{HERE}/models/fixed_volatility/missingobs.py")
fv_realdata = load("fv_realdata", f"{HERE}/models/fixed_volatility/realdata.py")


class nninputfun_fixvol(nninputfun):
    """Semiparametric repressilator with the volatility fixed at 0.1 (same drift as nninputfun)."""
    def g(self, t, y):
        return torch.ones_like(y) * 0.1


def semiparam(cls):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
    return cls(10 * torch.tensor([5., 5., 5.]), 10 * torch.tensor([1., 1., 1.]), 3.3 * torch.tensor([.01, .01, .01]), net=net, zero_init=False)


def pbmc(cls):
    torch.manual_seed(0); n = 30
    net = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n)).to(torch.float64)
    return cls(20 * torch.ones(n) * 5., 20 * torch.ones(n), np.sqrt(20) * torch.ones(n) * .01, net=net, zero_init=False)


class MLP(nn.Module):
    def __init__(self, dim, mid=[64]):
        super().__init__()
        layers, prev = [], dim
        for m in mid:
            layers += [nn.Linear(prev, m), nn.ReLU()]; prev = m
        layers.append(nn.Linear(prev, dim)); self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NNdrift(nn.Module):
    """Fully neural SDE of the ablation: MLP drift, diagonal constant volatility."""
    def __init__(self, net, sigma_vec):
        super().__init__()
        self.net = net; self.preprocesspos = torch.exp; self.sigma_vec = nn.Parameter(torch.log(sigma_vec))
        self.noise_type = "diagonal"; self.sde_type = "ito"

    def f(self, t, y):
        return self.net(y)

    def g(self, t, y):
        return torch.ones_like(y) * self.preprocesspos(self.sigma_vec)


LV = lambda m: m.LotkaVolterra(.5 * 9, .1 * 9, .1 * 9, .02 * 9, .01 * 3)
REP = lambda m: m.repressilator(10., 1., 1., 10., .03)
PROT = lambda m: m.repressilator(alpha=1e-5, beta_m=10., n=1., k=1., gamma_m=10., beta_p=10., gamma_p=10., sigma=0.09)
GOM = lambda m: m.lamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0., .01)
# method -> task -> (model builder, number of unobserved coordinates)
MODELS = {
    "Ours": {"LV": (lambda: LV(classic), 0), "ReprParam": (lambda: REP(classic), 0), "ReprSemiparam": (lambda: semiparam(nninputfun), 0),
             "ReprProtein": (lambda: PROT(missing), 3), "GoM": (lambda: GOM(realdata), 0), "PBMC": (lambda: pbmc(nninputfun), 0)},
    "Ours (mRNA-only model)": {"ReprProtein": (lambda: REP(classic), 0)},
    "Fixed volatility": {"LV": (lambda: LV(fv_classic), 0), "ReprParam": (lambda: REP(fv_classic), 0), "ReprSemiparam": (lambda: semiparam(nninputfun_fixvol), 0),
                         "ReprProtein": (lambda: PROT(fv_missing), 3), "GoM": (lambda: GOM(fv_realdata), 0), "PBMC": (lambda: pbmc(fv_realdata.nninputfunfixvol), 0)},
}
CKPT = {"Ours": "snapmmd", "Ours (mRNA-only model)": "snapmmd_mrna_only", "Fixed volatility": "fixed_volatility", "Fully neural": "fully_neural"}
NEURAL = {"LV": (2, 0, torch.float32), "ReprSemiparam": (3, 0, torch.float32), "ReprProtein": (6, 3, torch.float32), "GoM": (2, 0, torch.float32), "PBMC": (30, 0, torch.float64)}


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)


def save_figure_path(method, label, seed, traj, d):
    if seed == FIG_SEED_INTERP:
        out = f"{ROOT}/generated/paths_for_figures/{method}/{label}.npz"; os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, path=path_for_figure(traj, d))


def run_sde(method, label, seed, writer):
    data = np.load(f"{DATA}/{TASKS[label]}.npz"); val = np.load(f"{DATA}/{TASKS[label]}_interp_val.npz")
    N = int(data["N_steps"]); Xs = [torch.tensor(data["Xs"][i]) for i in range(N - 1)]
    dts = torch.tensor(data["dts"]); ts = torch.tensor(data["time_scale"]); d = Xs[0].shape[1]
    build, latent = MODELS[method][label]; model = build()
    model.load_state_dict(torch.load(f"{ROOT}/checkpoints/{CKPT[method]}/{label}/model_{seed}.pt", map_location="cpu"))
    X0 = Xs[0]
    if latent:
        X0 = torch.concatenate((X0, torch.zeros_like(X0)), axis=1)
    with torch.no_grad():
        seed_all(seed); fc = torchsde.sdeint(model, X0, torch.tensor([0, dts[-1] / ts]), method="euler")
        seed_all(seed + 1000); tr = torchsde.sdeint(model, X0, torch.linspace(dts[0] / ts, dts[-2] / ts, 500), method="euler")
        seed_all(seed + 2000)                                                    # R^2 as in snapMMD.train()
        n = torch.tensor([x.shape[0] for x in Xs], dtype=torch.float64); w = (n / n.sum()) ** 2; w = w / w.sum()
        rbf = RBF(); rbf.bandwidth = rbf.get_bandwidth_from_data(torch.cat(Xs)); M = MMDLoss(kernel=rbf); pooled = torch.cat(Xs)
        ssr = sum(w[i] * M(pooled, Xs[i]) for i in range(len(Xs)))
        ys = torchsde.sdeint(model, X0.repeat([5, 1]), dts[:-1] / ts, method="euler")
        r2 = 1. - sum(w[i] * M(ys[i][:, :d], Xs[i]) for i in range(len(Xs))) / ssr
    save_output(f"{ROOT}/generated/{method}/{label}/seed_{seed}.npz", d, forecast=fc[-1].numpy(), traj=tr.numpy(), n_val=val["Xs"].shape[0])
    save_figure_path(method, label, seed, tr.numpy(), d)
    if writer: writer.writerow([method, label, seed, f"{float(r2):.6f}"])


def run_neural(label, seed, writer):
    data = np.load(f"{DATA}/{TASKS[label]}.npz"); val = np.load(f"{DATA}/{TASKS[label]}_interp_val.npz")
    dim, latent, dt = NEURAL[label]; dts = data["dts"]; ts = float(data["time_scale"])
    X0 = torch.tensor(data["Xs"][0]).to(dt)
    if latent:
        X0 = torch.cat([X0, torch.zeros(X0.shape[0], latent, dtype=dt)], 1)
    model = NNdrift(MLP(dim), 3.3 * torch.tensor([.01] * dim)).to(dt)
    model.load_state_dict(torch.load(f"{ROOT}/checkpoints/fully_neural/{label}/model_{seed}.pt", map_location="cpu")); model = model.to(dt)
    with torch.no_grad():
        seed_all(seed); fc = torchsde.sdeint(model, X0, torch.tensor([0., float(dts[-1]) / ts], dtype=dt), method="euler")
        torch.manual_seed(seed + 1000); tr = torchsde.sdeint(model, X0, torch.linspace(float(dts[0]) / ts, float(dts[-2]) / ts, 500, dtype=dt), method="euler")
    d = data["Xs"].shape[-1]
    save_output(f"{ROOT}/generated/Fully neural/{label}/seed_{seed}.npz", d, forecast=fc[-1].numpy(), traj=tr.numpy(), n_val=val["Xs"].shape[0])
    save_figure_path("Fully neural", label, seed, tr.numpy(), d)
    if writer: writer.writerow(["Fully neural", label, seed, ""])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="*", default=list(MODELS) + ["Fully neural"]); ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int); a = ap.parse_args(); seeds = a.seeds or SEEDS
    os.makedirs(f"{ROOT}/generated", exist_ok=True)
    full = not a.tasks and not a.seeds and set(a.methods) == set(MODELS) | {"Fully neural"}   # a full run starts a fresh file; partial runs append
    with open(f"{ROOT}/generated/r2.csv", "w" if full else "a", newline="") as f:
        w = csv.writer(f) if not a.seeds else None
        for method in a.methods:
            labels = list(NEURAL) if method == "Fully neural" else list(MODELS[method])
            for label in labels:
                if a.tasks and label not in a.tasks: continue
                for s in seeds:
                    run_neural(label, s, w) if method == "Fully neural" else run_sde(method, label, s, w)
                print(method, label, "done", flush=True)
