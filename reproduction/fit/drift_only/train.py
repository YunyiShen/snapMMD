"""Drift-family fits with frozen or learned scalar state-dependent diffusion.

Uses the released structured models, training MMD, and evaluation metrics.
No test/interpolation targets enter the training loss or model selection.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torchsde

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "package"))
sys.path.insert(0, str(REPO / "reproduction/code"))
from snapMMD.dls import MMDLoss, RBF
from models.classic import LotkaVolterra, repressilator
from common import mmd2, emd
from misspecified import SaturatingLV, ExponentialRepressilator, training_saturation_scale

STEMS = {"LV": "LV", "ReprParam": "Repressilator"}
SEEDS = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
HISTORY_FIELDS = ["epoch", "loss_before_update", "grad_norm", "seconds", "learning_rate"]
MISSPECIFIED = {"affine", "lv_saturated", "repr_exponential"}


class FixedDiffusionSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, arm, time_scale):
        super().__init__()
        self.drift = drift
        self.arm = arm
        # Physical g(x)=0.02*x_+; under s=t/c the solver uses sqrt(c)*g(x).
        self.register_buffer("sigma_solver", torch.tensor(0.02 * math.sqrt(time_scale)))

    def f(self, t, y):
        return self.drift.f(t, y) if self.arm == "structured" else self.drift(y)

    def g(self, t, y):
        return self.sigma_solver * torch.relu(y)


class LearnedStateDiffusionSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift):
        super().__init__()
        self.drift = drift
        # The paper initializes sigma in SOLVER time, not physical time.
        # One learned scalar shared by all coordinates, as in the classic models.
        self.sigma = nn.Parameter(torch.log(torch.tensor(0.03)))

    def f(self, t, y):
        return self.drift(y)

    def g(self, t, y):
        return self.sigma.exp() * torch.relu(y)


def make_model(task, arm, time_scale, seed, diffusion="fixed_true", saturation_scale=None):
    if diffusion not in ("fixed_true", "learned_state"):
        raise ValueError("Unknown diffusion mode")
    if task not in STEMS or arm not in {"structured", "mlp"} | MISSPECIFIED:
        raise ValueError("Unknown task or drift family")
    if arm in MISSPECIFIED and diffusion != "learned_state":
        raise ValueError("Misspecified drift fits require --diffusion learned_state")
    if (arm == "lv_saturated" and task != "LV") or (arm == "repr_exponential" and task != "ReprParam"):
        raise ValueError("Drift family is incompatible with the task")
    if arm != "lv_saturated" and saturation_scale is not None:
        raise ValueError("Saturation scale only applies to lv_saturated")
    torch.manual_seed(seed)
    if arm == "structured":
        if task == "LV":
            drift = LotkaVolterra(0.5 * 9, 0.1 * 9, 0.1 * 9, 0.02 * 9, 0.03)
        else:
            drift = repressilator(10., 1., 1., 10., 0.03)
        if diffusion == "learned_state":
            return drift  # Use the original paper model, including its learned g.
        # The frozen wrapper bypasses the old g; never optimize its unused sigma.
        drift.sigma.requires_grad_(False)
    elif arm == "mlp":
        dim = 2 if task == "LV" else 3
        # Same architecture and initialization distribution as the old MLP;
        # use this run's seed instead of its historical hard-coded seed 0.
        drift = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim))
    elif arm == "affine":
        dim = 2 if task == "LV" else 3
        drift = nn.Linear(dim, dim)  # Raw signed states: no activation or clipping.
    elif arm == "lv_saturated":
        if saturation_scale is None:
            raise ValueError("Saturated LV requires a fixed training-derived scale")
        drift = SaturatingLV(saturation_scale, time_scale)
    else:
        drift = ExponentialRepressilator(time_scale)
    if diffusion == "learned_state":
        return LearnedStateDiffusionSDE(drift)
    return FixedDiffusionSDE(drift, arm, time_scale)


def sigma_solver(model):
    value = model.sigma_solver if isinstance(model, FixedDiffusionSDE) else model.sigma.exp()
    return float(value.detach())


def make_scheduler(optimizer, schedule, epochs, min_lr):
    if schedule == "constant":
        return None
    if schedule != "cosine":
        raise ValueError("Unknown schedule")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def gradient_norm(params, gradclip):
    if gradclip is not None:
        return torch.nn.utils.clip_grad_norm_(params, gradclip, error_if_nonfinite=True)
    # Observe the norm without modifying any gradient in the paper control.
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    return norm


def load_training(path, device="cpu"):
    with np.load(path, allow_pickle=False) as z:
        xs = torch.as_tensor(z["Xs"][:-1].copy(), dtype=torch.float32, device=device)
        physical_times = np.asarray(z["dts"][:-1], dtype=float)
        scale = float(z["time_scale"])
        assert len(xs) == int(z["N_steps"]) - 1
    times = torch.tensor(physical_times / scale, dtype=torch.float32, device=device)
    assert scale > 0 and torch.all(times[1:] > times[:-1]) and times[0] == 0
    return xs, times, scale


def make_brownian(y0, times, seed):
    # An explicit entropy and an order-independent tree make pairs and resumes
    # share paths. The old fitter relied on an implicit BrownianInterval seed.
    # Enclose float32 endpoints on the tree's rounded time grid; otherwise its
    # endpoint rounding can put the requested final time just outside the tree.
    tol = 1e-6
    return torchsde.BrownianInterval(
        t0=math.floor(float(times[0]) / tol) * tol,
        t1=math.ceil(float(times[-1]) / tol) * tol, size=y0.shape,
        dtype=y0.dtype, device=y0.device, entropy=seed,
        halfway_tree=True, tol=tol,
    )


def training_objective(xs):
    kernel = RBF().to(xs.device)
    kernel.bandwidth = kernel.get_bandwidth_from_data(torch.cat(list(xs)))
    if not torch.isfinite(kernel.bandwidth) or kernel.bandwidth <= 0:
        raise ValueError("Nonpositive or nonfinite training bandwidth")
    criterion = MMDLoss(kernel).to(xs.device)
    weights = torch.tensor([len(x) for x in xs], device=xs.device, dtype=xs.dtype)
    weights = (weights / weights.sum()).square()
    weights = weights / weights.sum()
    return criterion, weights


def objective(model, y0, times, xs, criterion, weights, bm, dt):
    ys = torchsde.sdeint_adjoint(model, y0, times, bm=bm, method="euler", dt=dt)
    return sum(w * criterion(y, x) for w, y, x in zip(weights, ys, xs))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_provenance():
    result = {}
    for name, command in [("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])]:
        proc = subprocess.run(["git", "-C", str(REPO)] + command,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        result[name] = proc.stdout.strip() if proc.returncode == 0 else None
    return result


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, model, optimizer, epoch, config, scheduler=None):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch, "config": config}, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, path, interp_path, xs, scale, seed, dt, out):
    with np.load(path, allow_pickle=False) as z:
        truth = z["Xs"][-1].copy()
        last_t, target_t = map(float, z["dts"][-2:])
    with np.load(interp_path, allow_pickle=False) as z:
        keep = np.asarray(z["dts"]) < last_t
        interp_t = np.asarray(z["dts"], float)[keep]
        interp_truth = z["Xs"][keep].copy()
    clouds = {}
    rows = []

    def simulate(initial, physical_times, noise_seed, protocol):
        ts = torch.tensor(physical_times / scale, dtype=initial.dtype, device=initial.device)
        bm = make_brownian(initial, ts, noise_seed)
        ys = torchsde.sdeint(model, initial, ts, method="euler", bm=bm, dt=dt)
        array = ys.cpu().numpy()
        clouds[protocol] = array[1:] if protocol == "interp" else array[-1]
        # Preserve every returned cloud before any finiteness/scoring failure.
        # If the solver itself raises, earlier protocols and the fitted checkpoint remain.
        np.savez_compressed(out / "predictions.partial.npz", **clouds, interp_times=interp_t)
        if not torch.isfinite(ys).all():
            raise FloatingPointError("Nonfinite evaluation trajectory: " + protocol)
        return array

    # Common noise between arms; independent streams for the three protocols.
    clouds["forecast_first"] = simulate(xs[0], np.array([0., target_t]), seed + 20000, "forecast_first")[-1]
    clouds["forecast_last"] = simulate(xs[-1], np.array([last_t, target_t]), seed + 30000, "forecast_last")[-1]
    clouds["interp"] = simulate(xs[0], np.r_[0., interp_t], seed + 40000, "interp")[1:]
    for protocol in ("forecast_first", "forecast_last"):
        rows.append({"protocol": protocol, "time": target_t,
                     "mmd2": float(mmd2(truth, clouds[protocol])["snap"]),
                     "w2": emd(truth, clouds[protocol])})
    for t, target, cloud in zip(interp_t, interp_truth, clouds["interp"]):
        rows.append({"protocol": "interp", "time": float(t),
                     "mmd2": float(mmd2(target, cloud)["snap"]), "w2": emd(target, cloud)})
    persistence = xs[-1].cpu().numpy()
    rows.append({"protocol": "persistence_last", "time": target_t,
                 "mmd2": float(mmd2(truth, persistence)["snap"]), "w2": emd(truth, persistence)})
    np.savez_compressed(out / "predictions.npz", **clouds, interp_times=interp_t)
    (out / "predictions.partial.npz").unlink()
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "time", "mmd2", "w2"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing a CPU fallback")
        # Keep float32 comparisons independent of the device's TF32 defaults.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    path = args.data_root / "classic" / (STEMS[args.task] + "_data.npz")
    interp_path = path.with_name(path.stem + "_interp_val.npz")
    xs, times, scale = load_training(path, args.device)
    saturation_scale = training_saturation_scale(xs) if args.drift == "lv_saturated" else None
    model = make_model(args.task, args.drift, scale, args.seed, args.diffusion,
                       saturation_scale=saturation_scale).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = make_scheduler(optimizer, args.schedule, args.epochs, args.min_lr)
    y0 = xs[0].repeat(5, 1)
    criterion, weights = training_objective(xs)
    bm = make_brownian(y0, times, args.seed + 10000)
    source_paths = [Path(__file__).resolve(), REPO / "package/snapMMD/dls.py",
                    REPO / "reproduction/code/models/classic.py", REPO / "reproduction/code/common.py"]
    if args.drift in MISSPECIFIED:
        source_paths.append(Path(__file__).resolve().with_name("misspecified.py"))
    config = {
        "task": args.task, "drift": args.drift, "seed": args.seed,
        "epochs_policy": "fixed budget; no target-based selection", "lr": args.lr,
        "schedule": args.schedule,
        "schedule_epochs": args.epochs if scheduler is not None else None,
        "min_lr": args.min_lr if scheduler is not None else None,
        "dt_solver": args.dt, "gradclip": args.gradclip, "time_scale": scale,
        "diffusion_mode": args.diffusion, "diffusion": "diagonal relu(x)",
        "sigma_parameterization": "fixed" if args.diffusion == "fixed_true" else "learned positive scalar; log parameter",
        "sigma_solver_initial": sigma_solver(model),
        "sigma_physical_initial": sigma_solver(model) / math.sqrt(scale),
        "train_particles": len(y0),
        "train_times_solver": times.cpu().tolist(), "train_bandwidth": float(criterion.kernel.bandwidth),
        "trainable_parameters": sum(p.numel() for p in params), "dtype": "float32",
        "device": args.device, "threads": args.threads,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "tf32": False if args.device == "cuda" else None,
        "data_sha256": sha256(path), "interp_sha256": sha256(interp_path),
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in source_paths},
        "brownian": {"train_entropy": args.seed + 10000, "halfway_tree": True, "tol": 1e-6},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchsde": torchsde.__version__, "numpy": np.__version__},
    }
    if args.drift in MISSPECIFIED:
        config["drift_initial_state"] = {name: value.detach().cpu().tolist()
                                          for name, value in model.drift.state_dict().items()}
        config["drift_parameter_units"] = "solver-time rates; state scales unchanged"
        config["saturation_scale"] = saturation_scale
        config["saturation_scale_rule"] = "median of all training prey coordinates" if saturation_scale is not None else None
    out = args.output.resolve()
    epoch0 = 0
    if args.resume:
        state = torch.load(out / "checkpoint.pt", map_location=args.device)
        if state["config"] != config:
            raise ValueError("Resume configuration/source/data/environment mismatch")
        if (out / "summary.json").exists():
            raise ValueError("Completed output is immutable; use a new output directory")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            if state.get("scheduler") is None:
                raise ValueError("Missing scheduler state")
            scheduler.load_state_dict(state["scheduler"])
        elif state.get("scheduler") is not None:
            raise ValueError("Unexpected scheduler state")
        epoch0 = state["epoch"]
        if epoch0 > args.epochs:
            raise ValueError("Requested budget is below the checkpoint's epoch")
        # Drop log rows after the last durable checkpoint, preserving an audit copy.
        log = out / "history.csv"
        if log.exists():
            old = log.read_text()
            (out / ("history_before_resume_%d.csv" % time.time_ns())).write_text(old)
            rows = list(csv.DictReader(old.splitlines()))
            with log.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows(r for r in rows if int(r["epoch"]) <= epoch0)
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "config.json", config)
        save_checkpoint(out / "checkpoint.pt", model, optimizer, 0, config, scheduler)
    with (out / "invocations.jsonl").open("a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "command": sys.argv, "host": platform.node(),
                            "git": git_provenance(),
                            "job_id": os.environ.get("SLURM_JOB_ID"), "epoch_start": epoch0,
                            "epoch_target": args.epochs}) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    epoch_seconds = []
    try:
        with (out / "history.csv").open("a", newline="", buffering=1) as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(HISTORY_FIELDS)
            for epoch in range(epoch0 + 1, args.epochs + 1):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad()
                loss = objective(model, y0, times, xs, criterion, weights, bm, args.dt)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss at epoch %d" % epoch)
                loss.backward()
                norm = gradient_norm(params, args.gradclip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if not all(torch.isfinite(p).all() for p in params):
                    raise FloatingPointError("Nonfinite parameters at epoch %d" % epoch)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                epoch_seconds.append(seconds)
                writer.writerow([epoch, float(loss), float(norm), seconds, lr_used])
                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print("epoch=%d loss=%.7g lr=%.7g sigma_solver=%.7g seconds=%.3f" %
                          (epoch, loss, lr_used, sigma_solver(model), seconds), flush=True)
                if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                    save_checkpoint(out / "checkpoint.pt", model, optimizer, epoch, config, scheduler)
        training_seconds = time.perf_counter() - start
        # A fresh Brownian tree with the same entropy also verifies end-of-fit fit,
        # rather than mislabeling the last pre-update loss as the final objective.
        with torch.no_grad():
            ys = torchsde.sdeint(model, y0, times, method="euler", dt=args.dt,
                                bm=make_brownian(y0, times, args.seed + 10000))
            final_loss = sum(w * criterion(y, x) for w, y, x in zip(weights, ys, xs))
        summary = {"epochs": args.epochs, "epochs_this_invocation": args.epochs - epoch0,
                   "training_seconds_this_invocation": training_seconds,
                   "median_epoch_seconds_after_5_warmup": float(np.median(epoch_seconds[5:])) if len(epoch_seconds) > 5 else None,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                   "final_training_loss": float(final_loss),
                   "sigma_solver_final": sigma_solver(model),
                   "sigma_physical_final": sigma_solver(model) / math.sqrt(scale),
                   "learning_rate_after_final_update": optimizer.param_groups[0]["lr"]}
        write_json(out / "training_summary.json", summary)
        metrics = evaluate(model, path, interp_path, xs, scale, args.seed, args.dt, out)
        summary["interp_mean_mmd2"] = float(np.mean([r["mmd2"] for r in metrics if r["protocol"] == "interp"]))
        write_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(out / ("failure_%d.json" % time.time_ns()),
                   {"type": type(error).__name__, "message": str(error)})
        raise


def optional_clip(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use a positive norm bound or 'none'")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=STEMS, required=True)
    p.add_argument("--drift", choices=["structured", "mlp", "affine", "lv_saturated", "repr_exponential"], required=True)
    p.add_argument("--diffusion", choices=["fixed_true", "learned_state"], default="fixed_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "data")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--min-lr", type=float, default=0.0001)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--gradclip", type=optional_clip, default=1., help="positive norm bound, or 'none'")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    for key in ["epochs", "lr", "dt", "threads", "checkpoint_every", "log_every"]:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error("--%s must be positive" % key.replace("_", "-"))
    if args.gradclip is not None and (not math.isfinite(args.gradclip) or args.gradclip <= 0):
        p.error("--gradclip must be positive or 'none'")
    if not math.isfinite(args.min_lr) or args.min_lr < 0:
        p.error("--min-lr must be finite and nonnegative")
    if args.schedule == "cosine" and args.min_lr >= args.lr:
        p.error("cosine --min-lr must be smaller than --lr")
    if args.drift in MISSPECIFIED and args.diffusion != "learned_state":
        p.error("misspecified drifts require --diffusion learned_state")
    if (args.drift == "lv_saturated" and args.task != "LV") or (args.drift == "repr_exponential" and args.task != "ReprParam"):
        p.error("drift family is incompatible with --task")
    return args


if __name__ == "__main__":
    run(parse_args())
