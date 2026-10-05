"""Campaign 4: run PRESCIENT, PI-SDE, scNODE and JKOnet* on the paper's snapshot data.

One call fits one (method, variant, task, seed) and writes, in the layout of the other baselines,
  results/outputs/{method}_{variant}/{task}_forecast_{seed}.npz        'forecast': (2, N, d)   [first snapshot, forecast at t_T]
  results/outputs/{method}_{variant}/{task}_interpolation_{seed}.npz   'interpolation': (n_steps, N, d) on the simulation grid
                                                                          from t_1 to t_{T-1} (step 0.1 in the baseline's time unit)
  results/outputs/{method}_{variant}/{task}_log_{seed}.json            settings, training losses, wall time

Faithfulness: the four training loops are the authors' (PRESCIENT train/run.py; PI-SDE src/train.py;
scNODE optim/running.py), with their default hyperparameters, reproduced here on in-memory arrays
instead of their data files. Departures, all forced by our data and all recorded in the log:
  - time unit: the data's own time grid (PBMC in hours), so that the authors' step of 0.1 gives
    8-10 Euler steps per interval;
  - noise scale (PRESCIENT train_sd, PI-SDE sigma_const): 0.1 and 0.5 are run as variants;
  - PI-SDE is integrated with torchsde.sdeint (its own adjoint solver is a copy of torchsde's);
  - scNODE: latent dimension = data dimension, linear encoder/decoder (its ReLU decoder cannot
    output the negative GoM coordinates), ODE solved with dopri5 (its euler option takes one step per
    requested time, which is far too coarse for oscillating systems);
  - JKOnet* (added 2026-10-04): the authors' model classes, loss, optimiser and one-step-per-interval predictor
    (vendor/jkonet_star), on couplings and densities built as their data_generator.py builds them; see run_jkonet
    for the one constant that had to change (the density estimator's determinant filter) and for the fine-grid
    interpolation rule. Runs in the `jkonet-star` conda env (JAX).
Every method starts from the observed first snapshot and sees the training snapshots only.

Usage: python run_baselines.py --method prescient|pisde|scnode|jkonet --task LV --seed 42 [--variant sd0.5] [--smoke]
"""
import argparse
import json
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))          # snapMMD/data
OUT = f"{HERE}/results/outputs"
torch.set_num_threads(1)

TASKS = {  # label: (data file, time unit factor applied to dts)
    "LV": ("classic/LV_data", 1.0),
    "ReprParam": ("classic/Repressilator_data", 1.0),
    "ReprProtein": ("missingobs/Repressilator_data", 1.0),
    "GoM": ("realdata/GoM_data", 1.0),
    "PBMC": ("realdata/processed_pbmc_data_sub500_every_2_until20", 20.0),   # hours
}
# ReprSemiparam uses the same data as ReprParam; the baselines have no model family, so one run serves both rows.
DT = 0.1


def load_task(label):
    f, unit = TASKS[label]
    d = np.load(f"{DATA}/{f}.npz")
    times = (d["dts"] * unit).astype(np.float64)
    snaps = [torch.tensor(np.asarray(x, dtype=np.float32)) for x in d["Xs"]]
    return SimpleNamespace(label=label, times=times, train=snaps[:-1], train_times=times[:-1],
                           forecast_time=float(times[-1]), x0=snaps[0], N=snaps[0].shape[0], d=snaps[0].shape[1])


def grid_times(task):
    """simulation grid: t_1 to t_{T-1} in steps of DT, then the forecast time appended."""
    t0, t_last, t_fc = task.train_times[0], task.train_times[-1], task.forecast_time
    n = int(round((t_last - t0) / DT))
    g = t0 + DT * np.arange(n + 1)
    return g, np.append(g, t_fc)


def save(method, variant, task, seed, traj, forecast, log, **extra):
    out = f"{OUT}/{method}_{variant}"
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/{task.label}_interpolation_{seed}.npz", interpolation=traj.astype(np.float32))
    np.savez(f"{out}/{task.label}_forecast_{seed}.npz", forecast=forecast.astype(np.float32),
             **{k: np.asarray(v).astype(np.float32) for k, v in extra.items()})
    json.dump(log, open(f"{out}/{task.label}_log_{seed}.json", "w"), indent=1)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- PRESCIENT
def run_prescient(task, seed, sd, smoke):
    sys.path.insert(0, f"{HERE}/vendor")
    from prescient_train.model import AutoGenerator, OTLoss
    from prescient_train.util import p_samp, fit_regularizer
    from torch import optim
    seed_all(seed)
    x, y = task.train, [float(t) for t in task.train_times]
    T = len(x)
    cfg = SimpleNamespace(   # prescient/commands/train_model.py init_config, defaults
        x_dim=task.d, k_dim=500, layers=1, activation="softplus",
        pretrain_burnin=50, pretrain_sd=0.1, pretrain_lr=1e-9, pretrain_epochs=50 if smoke else 500,
        train_dt=DT, train_sd=sd, train_batch=0.1, ns=2000, train_burnin=100, train_tau=1e-6,
        train_epochs=50 if smoke else 2500, train_lr=0.01, train_clip=0.25,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, out_name=f"{task.label}-{seed}",
        t=y[-1] - y[0], start_t=0, train_t=list(range(1, T)))
    device = torch.device("cpu")
    model = AutoGenerator(cfg)
    loss = OTLoss(cfg, device)
    t_start = time.time()
    # pretraining (contrastive divergence from the last training snapshot), as in run.py
    x_last = x[cfg.train_t[-1]]
    optimizer = optim.SGD(list(model.parameters()), lr=cfg.pretrain_lr)
    for epoch in range(cfg.pretrain_epochs):
        if epoch % 100 == 0:
            print(f"[pretrain] epoch {epoch} elapsed {time.time() - t_start:.0f}s", flush=True)
        pp, _ = p_samp(x_last, cfg.ns)
        pp, pos_fv, neg_fv = fit_regularizer(x_last, pp, cfg.pretrain_burnin, cfg.t / cfg.pretrain_burnin, cfg.pretrain_sd, model, device)
        (pos_fv + neg_fv).backward()
        optimizer.step()
        model.zero_grad()
    # training, as in run.py (uniform growth weights)
    optimizer = optim.Adam(list(model.parameters()), lr=cfg.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    optimizer.zero_grad()
    losses = []
    for epoch in range(cfg.train_epochs):
        losses_xy = []
        for j in cfg.train_t:
            dat_prev, dat_cur = x[cfg.start_t], x[j]
            time_elapsed = y[j] - y[cfg.start_t]
            x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * cfg.train_batch), None)
            for _ in range(int(np.round(time_elapsed / cfg.train_dt))):
                z = torch.randn(x_i.shape[0], x_i.shape[1]) * cfg.train_sd
                x_i = model._step(x_i, dt=cfg.train_dt, z=z)
            y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * cfg.train_batch))
            loss_xy = loss(a_i, x_i, b_j, y_j)
            losses_xy.append(loss_xy.item())
            loss_xy.backward()
        if cfg.train_tau > 0:
            pp, _ = p_samp(x_last, cfg.ns)
            pp, pos_fv, neg_fv = fit_regularizer(x_last, pp, cfg.train_burnin, cfg.t / cfg.train_burnin, cfg.train_sd, model, device)
            ((pos_fv + neg_fv) * cfg.train_tau).backward()
        if cfg.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train_clip)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        losses.append(float(np.mean(losses_xy)))
        if epoch % 100 == 0:
            print(f"[train] epoch {epoch} loss {losses[-1]:.4f} elapsed {time.time() - t_start:.0f}s", flush=True)
    train_time = time.time() - t_start
    # simulate from the observed first snapshot (all cells) on the 0.1 grid up to the forecast time
    seed_all(seed + 1000)
    g, g_fc = grid_times(task)
    n_total = int(round((task.forecast_time - y[0]) / DT))
    xs = [task.x0.clone()]
    xi = task.x0.clone()
    for _ in range(n_total):
        z = torch.randn(xi.shape[0], xi.shape[1]) * cfg.train_sd
        xi = model._step(xi, dt=cfg.train_dt, z=z).detach()
        xs.append(xi.clone())
    xs = torch.stack(xs).numpy()                      # (n_total+1, N, d) at t0 + k*DT
    traj = xs[:len(g)]
    forecast = np.stack([xs[0], xs[-1]])
    log = dict(method="prescient", variant=f"sd{sd}", config={k: v for k, v in vars(cfg).items()}, losses=losses,
               train_time_s=train_time, grid_times=g.tolist(), forecast_time=task.forecast_time)
    return traj, forecast, log


# ---------------------------------------------------------------- PI-SDE
def run_pisde(task, seed, sigma_type, sigma_const, smoke):
    sys.path.insert(0, f"{HERE}/vendor")
    from pisde.model import AutoGenerator
    from geomloss import SamplesLoss
    import torchsde
    from torch import optim
    seed_all(seed)
    x, y = task.train, [float(t) for t in task.train_times]
    T = len(x)
    cfg = SimpleNamespace(   # PI-SDE src/config_Veres.py defaults
        x_dim=task.d, k_dims=[400, 400], layers=2, activation="softplus", sigma_type=sigma_type, sigma_const=sigma_const,
        train_epochs=50 if smoke else 3000, train_lr=0.005, train_lambda=0.5, train_batch=0.1, train_clip=0.1,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, start_t=0, train_t=list(range(1, T)))
    func = AutoGenerator(cfg)
    ot = SamplesLoss("sinkhorn", p=2, blur=cfg.sinkhorn_blur, scaling=cfg.sinkhorn_scaling, debias=True)

    def p_samp(p, n):
        idx = np.random.choice(p.shape[0], size=n, replace=p.shape[0] < n)
        w = torch.ones(n) / n
        return p[idx, :].clone(), w

    def integrate(x_r_0, ts):
        return torchsde.sdeint(func, x_r_0, torch.tensor(ts, dtype=torch.float32), method="euler", dt=DT,
                               names={"drift": "f", "diffusion": "g"})

    optimizer = optim.Adam(list(func.parameters()), lr=cfg.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    optimizer.zero_grad()
    t_start = time.time()
    losses, losses_r = [], []
    ts = [y[cfg.start_t]] + [y[j] for j in cfg.train_t]
    for epoch in range(cfg.train_epochs):
        dat_prev = x[cfg.start_t]
        nb = int(dat_prev.shape[0] * cfg.train_batch)
        x_i, a_i = p_samp(dat_prev, nb)
        x_r_i = torch.cat([x_i, torch.zeros(nb, 1)], dim=1)
        x_r_s = integrate(x_r_i, ts)
        lxy, lr_ = [], []
        for pos, j in enumerate(cfg.train_t):
            y_j, b_j = p_samp(x[j], int(x[j].shape[0] * cfg.train_batch))
            loss_xy = ot(a_i, x_r_s[pos + 1][:, :-1], b_j, y_j)
            lxy.append(loss_xy.item())
            if cfg.train_lambda > 0 and j == cfg.train_t[-1]:
                loss_r = torch.mean(x_r_s[-1][:, -1] * cfg.train_lambda)
                lr_.append(loss_r.item())
                loss_all = loss_xy + loss_r
            else:
                loss_all = loss_xy
            loss_all.backward(retain_graph=True)
        if cfg.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(func.parameters(), cfg.train_clip)
        optimizer.step()
        scheduler.step()
        func.zero_grad()
        losses.append(float(np.mean(lxy)))
        losses_r.append(float(np.mean(lr_)) if lr_ else 0.0)
        if epoch % 100 == 0:
            print(f"[train] epoch {epoch} loss {losses[-1]:.4f} hj {losses_r[-1]:.4f} elapsed {time.time() - t_start:.0f}s", flush=True)
    train_time = time.time() - t_start
    seed_all(seed + 1000)
    g, g_fc = grid_times(task)
    with torch.no_grad():
        pass
    x_r_0 = torch.cat([task.x0.clone(), torch.zeros(task.N, 1)], dim=1)
    xs = integrate(x_r_0, g_fc.tolist()).detach().numpy()[:, :, :-1]    # (len(g)+1, N, d)
    traj, forecast = xs[:len(g)], np.stack([xs[0], xs[-1]])
    log = dict(method="pisde", variant=f"{sigma_type}{sigma_const if sigma_type == 'const' else ''}",
               config={k: v for k, v in vars(cfg).items()}, losses=losses, losses_hj=losses_r,
               train_time_s=train_time, grid_times=g.tolist(), forecast_time=task.forecast_time)
    return traj, forecast, log


# ---------------------------------------------------------------- scNODE
def run_scnode(task, seed, smoke):
    sys.path.insert(0, f"{HERE}/vendor/scnode")
    from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
    # scNODE builds dist.Normal(mu, std) for a return value that neither the loss nor the sampling uses;
    # torch rejects an exact zero std there (seen on ReprProtein seeds 42 and 43). Validation off, nothing else changes.
    torch.distributions.Distribution.set_default_validate_args(False)
    seed_all(seed)
    # scNODE works in index time; our training times are equally spaced, so index = (t - t0)/spacing
    spacing = float(task.train_times[1] - task.train_times[0])
    to_idx = lambda t: (np.asarray(t) - task.train_times[0]) / spacing
    train_tps = torch.FloatTensor(to_idx(task.train_times))
    cfg = dict(latent_dim=task.d, drift_latent_size=[64], enc_latent_list=None, dec_latent_list=None,
               latent_enc_act="none", latent_dec_act="none", drift_act="relu", ode_method="dopri5",
               latent_coeff=1.0, epochs=2 if smoke else 10, iters=20 if smoke else 100, batch_size=32, lr=1e-3,
               pretrain_iters=200, pretrain_lr=1e-3)
    model = constructscNODEModel(task.d, latent_dim=cfg["latent_dim"], enc_latent_list=cfg["enc_latent_list"],
                                 dec_latent_list=cfg["dec_latent_list"], drift_latent_size=cfg["drift_latent_size"],
                                 latent_enc_act=cfg["latent_enc_act"], latent_dec_act=cfg["latent_dec_act"],
                                 drift_act=cfg["drift_act"], ode_method=cfg["ode_method"])
    t_start = time.time()
    model, loss_list, _, _, _ = scNODETrainWithPreTrain(
        task.train, train_tps, model, latent_coeff=cfg["latent_coeff"], epochs=cfg["epochs"], iters=cfg["iters"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], pretrain_iters=cfg["pretrain_iters"], pretrain_lr=cfg["pretrain_lr"])
    train_time = time.time() - t_start
    seed_all(seed + 1000)
    g, g_fc = grid_times(task)
    pred = scNODEPredict(model, task.x0, torch.FloatTensor(to_idx(g_fc)), n_cells=task.N)   # (N, n_tps, d)
    xs = np.moveaxis(pred, 0, 1)
    traj, forecast = xs[:len(g)], np.stack([xs[0], xs[-1]])
    log = dict(method="scnode", variant="default", config=cfg, losses=[l[0] for l in loss_list],
               train_time_s=train_time, grid_times=g.tolist(), forecast_time=task.forecast_time)
    return traj, forecast, log


# ---------------------------------------------------------------- JKOnet*
JKONET_SOLVERS = {"full": "jkonet-star",                        # potential + internal (diffusion) + interaction energy
                  "potential": "jkonet-star-potential",         # the default solver of the authors' train.py
                  "nointer": "jkonet-star-potential-internal"}  # potential + diffusion


def run_jkonet(task, seed, variant, smoke):
    """JKOnet* (Terpin, Lanzetti, Gadea, Dorfler, NeurIPS 2024). Everything that defines the method is the authors'
    code in vendor/jkonet_star: model classes (loss, networks, optimiser, train_step), the exact-OT couplings between
    consecutive snapshots (pairs with plan weight > 1/(10 N)), the 10-component Gaussian mixture per snapshot that
    gives the density and its gradient, and the predictor (one explicit step per snapshot interval, SDESimulator).
    Their config.yaml defaults: 1000 epochs, batches of 250 couplings, Adam lr 1e-3, clip 10, MLP [64, 64] softplus.
    The step tau is the snapshot spacing in the data's time unit (their code hard-codes dt = 1 between snapshots).

    One departure, forced by our data: GaussianMixtureModel.fit drops every mixture component whose covariance
    determinant is below 1e-4, an absolute threshold suited to their synthetic data spread over [-4, 4]^2. Our
    snapshots are tight (spread 0.03 at t_1, below 1 afterwards), so the filter removes every component at nearly
    every time point; the density is then the clip value 1e-5 with zero gradient, the diffusion term gets no
    gradient and beta stays at its random initial value. The threshold is set to 0 here (sklearn's reg_covar keeps
    the covariances non-singular); the number of components kept is in the log.

    Outputs: 'forecast' = the authors' predictor, T steps of size tau from the observed first snapshot (the last one
    lands on the forecast time); 'interpolation' = that path interpolated linearly in time onto the DT grid, i.e. each
    particle moves in a straight line at constant speed between two steps of the scheme (the geodesic interpolation
    of a JKO scheme; for the Brownian part it has the exact variance at the midpoints, where the metric reads it).
    Simulating the Langevin SDE dX = -grad V dt + sqrt(2 beta) dW with a finer step is not an option: the one-step fit
    absorbs the step, and grad V is unconstrained away from the snapshot points (checked on LV: the fine simulation
    leaves the data at once, MMD^2 1.4 at every validation time against 0.6-0.9 for the scheme's own steps).
    'forecast_native_path' = the whole path of the predictor, (T+1, N, d). The authors' one-step-ahead protocol:
    'forecast_onestep' = one step from the observed last training snapshot, (N, d); 'interpolation_onestep' = within
    each interval, the straight line from the observed snapshot to its one-step prediction, on the DT grid. All extra
    arrays live in the forecast npz. The trained parameters are pickled next to them ({task}_params_{seed}.pkl)."""
    sys.path.insert(0, f"{HERE}/vendor/jkonet_star")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import jax
    import jax.numpy as jnp
    import yaml
    from models import EnumMethod, get_model
    from utils import density as density_module
    from utils.density import GaussianMixtureModel
    from utils.ot import compute_couplings
    from utils.sde_simulator import SDESimulator
    density_module.DET_THRESHOLD = 0.0
    seed_all(seed)
    V = f"{HERE}/vendor/jkonet_star"
    config = yaml.safe_load(open(f"{V}/config.yaml"))
    config.update(yaml.safe_load(open(f"{V}/config-jkonet-extra.yaml")))
    epochs = 5 if smoke else config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    n_gmm = 10                                                    # data_generator.py --n-gmm-components default
    tau = float(task.train_times[1] - task.train_times[0])
    solver = EnumMethod(JKONET_SOLVERS[variant])
    snaps = {i: jnp.asarray(x.numpy()) for i, x in enumerate(task.train)}
    T = len(snaps)
    # couplings and densities, as data_generator.generate_data_from_trajectory (balanced data, one batch)
    t_start = time.time()
    gmm = GaussianMixtureModel()
    gmm.fit(snaps, n_gmm, seed)
    kept = [int(m.shape[0]) for m in gmm.gms_means]
    rows, dens = [], []
    for t in range(T - 1):
        c = compute_couplings(snaps[t], snaps[t + 1], t + 1)
        ys = c[:, task.d:2 * task.d]
        rho_t1 = (lambda t1: (lambda x: gmm.gmm_density(t1, x)))(t + 1)
        dens.append(jnp.concatenate([jax.vmap(rho_t1)(ys).reshape(-1, 1), jax.vmap(jax.grad(rho_t1))(ys)], axis=1))
        rows.append(c)
    c, dens = jnp.concatenate(rows), jnp.concatenate(dens)
    xs, ys, ts, ws = c[:, :task.d], c[:, task.d:2 * task.d], c[:, -2], c[:, -1]
    rho, rho_grad = dens[:, 0], dens[:, 1:]
    n = int(xs.shape[0])
    prep_time = time.time() - t_start
    print(f"[prep] {n} couplings, gmm components kept per snapshot {kept}, {prep_time:.0f}s", flush=True)
    # training, as train.py (DataLoader shuffle replaced by a seeded permutation)
    model = get_model(solver, config, task.d, tau)
    state = model.create_state(jax.random.PRNGKey(seed))
    train_step = jax.jit(model.train_step)
    rng = np.random.RandomState(seed)
    losses = []
    t_start = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            l, state = train_step(state, (xs[idx], ys[idx], ts[idx], ws[idx], rho[idx], rho_grad[idx]))
            tot, nb = tot + float(l), nb + 1
        losses.append(tot / nb)
        if epoch % 100 == 0:
            print(f"[train] epoch {epoch} loss {losses[-1]:.4f} beta {model.get_beta(state):.4g} "
                  f"elapsed {time.time() - t_start:.0f}s", flush=True)
    train_time = time.time() - t_start
    # prediction
    potential, beta, interaction = model.get_potential(state), float(model.get_beta(state)), model.get_interaction(state)
    key_pred = jax.random.PRNGKey(seed + 1000)
    x0 = jnp.asarray(task.x0.numpy())
    native = np.asarray(SDESimulator(tau, T, 1, potential, beta, interaction).forward_sampling(key_pred, x0))  # (T+1, N, d)
    g, g_fc = grid_times(task)
    s_grid = (g - task.train_times[0]) / tau                      # position on the grid in units of steps
    k = np.minimum(np.floor(s_grid + 1e-6).astype(int), T - 1)
    w = (s_grid - k)[:, None, None]
    traj = (1 - w) * native[k] + w * native[k + 1]
    forecast = np.stack([np.asarray(x0), native[-1]])
    # The authors' own evaluation protocol (dataset.error_wasserstein_one_step_ahead): one step of the predictor from
    # the OBSERVED snapshot at t_k, compared with the snapshot at t_{k+1}. Saved for every interval: the step from the
    # last training snapshot lands on the forecast time ('forecast_onestep'); within each interval the particles move
    # in a straight line from the observed snapshot to its one-step prediction ('interpolation_onestep', DT grid).
    one = SDESimulator(tau, 1, 1, potential, beta, interaction)
    X_obs = np.stack([x.numpy() for x in task.train])                                              # (T, N, d)
    steps = np.stack([np.asarray(one.forward_sampling(jax.random.PRNGKey(seed + 2000 + k), jnp.asarray(X_obs[k])))[-1]
                      for k in range(T)])                                                          # (T, N, d)
    interp_onestep = (1 - w) * X_obs[k] + w * steps[k]
    extra = dict(forecast_native_path=native, forecast_onestep=steps[-1], interpolation_onestep=interp_onestep)
    # trained parameters, so that any other prediction protocol can be computed without re-fitting
    out_dir = f"{OUT}/jkonet_{variant}" + ("_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{task.label}_params_{seed}.pkl", "wb") as fh:
        pickle.dump(dict(solver=str(solver), tau=tau, beta=beta, layers=config["energy"]["model"]["layers"],
                         params=jax.tree_util.tree_map(np.asarray, model.get_params(state))), fh)
    log = dict(method="jkonet", variant=variant,
               config=dict(solver=str(solver), epochs=epochs, batch_size=batch_size, optim=config["energy"]["optim"],
                           layers=config["energy"]["model"]["layers"], tau=tau, n_couplings=n, gmm_components=n_gmm,
                           gmm_det_threshold=0.0, gmm_components_kept=kept, data="raw coordinates",
                           interpolation="piecewise-linear in time between the predictor's steps"),
               losses=losses, beta=beta, train_time_s=train_time, prep_time_s=prep_time,
               grid_times=g.tolist(), forecast_time=task.forecast_time)
    print(f"[pred] beta {beta:.4g}", flush=True)
    return traj, forecast, log, extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["prescient", "pisde", "scnode", "jkonet"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    task = load_task(a.task)
    extra = {}
    if a.method == "prescient":
        sd = float(a.variant.replace("sd", "")) if a.variant else 0.5
        traj, forecast, log = run_prescient(task, a.seed, sd, a.smoke)
        variant = f"sd{sd}"
    elif a.method == "pisde":
        v = a.variant or "const0.1"
        if v == "mlp":
            traj, forecast, log = run_pisde(task, a.seed, "Mlp", None, a.smoke)
        else:
            traj, forecast, log = run_pisde(task, a.seed, "const", float(v.replace("const", "")), a.smoke)
        variant = v
    elif a.method == "jkonet":
        variant = a.variant or "full"
        traj, forecast, log, extra = run_jkonet(task, a.seed, variant, a.smoke)
    else:
        traj, forecast, log = run_scnode(task, a.seed, a.smoke)
        variant = "default"
    log["smoke"] = a.smoke
    save(a.method, variant + ("_smoke" if a.smoke else ""), task, a.seed, traj, forecast, log, **extra)
    print(f"{a.method} {variant} {a.task} seed {a.seed}: train {log['train_time_s']:.0f}s, final loss {log['losses'][-1]:.4f}, "
          f"traj {traj.shape}, forecast {forecast.shape}, finite {np.isfinite(traj).all() and np.isfinite(forecast).all()}")
