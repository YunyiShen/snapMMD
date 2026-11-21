"""Evaluate saved coefficients and physical drift/covariance; never train or overwrite.

Run from any directory with the reproduction environment:
  python parameter_recovery.py --out /path/to/new/results-directory
The reported spread is over saved fits on one dataset, not dataset uncertainty.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common import DATA, ROOT, SEEDS, TASKS, Task, emd, mmd2
from simulate import MODELS

TRUTH = {
    "LV": dict(alpha=1., beta=.4, gamma=.4, delta=.1, sigma=.02),
    "ReprParam": dict(beta=10., n=3., k=1., gamma=1., sigma=.02),
    "ReprProtein": dict(alpha=1e-5, beta_m=10., beta_p=1., n=3., k=1.,
                        gamma_m=1., gamma_p=1., sigma=.02),
}
SCALE = {"LV": 9., "ReprParam": 10., "ReprProtein": 10., "ReprSemiparam": 10.}


def power(name):
    return .5 if name == "sigma" else (0. if name in ("n", "k") else 1.)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update((key + str(a.shape) + str(a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing empty table: " + str(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def reference_functions(task, x):
    """Physical generator equations, including the original positivity offsets.

    Source: Supercloud/{classic,missingobs}/make_data.py and models.py.
    Linear multiplicative diffusion has diagonal covariance .02**2 * relu(x)**2.
    """
    p = TRUTH["ReprParam" if task == "ReprSemiparam" else task]
    y = torch.relu(x) + (1e-10 if task == "LV" else 1e-8)
    if task == "LV":
        drift = torch.stack((p["alpha"] * y[:, 0] - p["beta"] * y[:, 0] * y[:, 1],
                             p["delta"] * y[:, 0] * y[:, 1] - p["gamma"] * y[:, 1]), 1)
    elif task in ("ReprParam", "ReprSemiparam"):
        drift = p["beta"] / (1 + (y[:, [2, 0, 1]] / p["k"]) ** p["n"]) - p["gamma"] * y
    else:
        dm = p["alpha"] + p["beta_m"] / (1 + (y[:, [5, 3, 4]] / p["k"]) ** p["n"]) - p["gamma_m"] * y[:, :3]
        dp = p["beta_p"] * y[:, :3] - p["gamma_p"] * y[:, 3:]
        drift = torch.cat((dm, dp), 1)
    return drift, (.02 * torch.relu(x)).square()


def check_conversion(task, model):
    c = SCALE[task]
    physical = copy.deepcopy(model)
    for name in TRUTH[task]:
        getattr(physical, name).sub_(power(name) * math.log(c))
    d = 2 if task == "LV" else (6 if task == "ReprProtein" else 3)
    x = torch.arange(1, 5 * d + 1, dtype=torch.float64).reshape(5, d) / 7
    ferr = float((physical.f(0., x) - model.f(0., x) / c).abs().max())
    gerr = float((physical.g(0., x) - model.g(0., x) / math.sqrt(c)).abs().max())
    assert torch.allclose(physical.f(0., x), model.f(0., x) / c, atol=1e-12, rtol=1e-12)
    assert torch.allclose(physical.g(0., x), model.g(0., x) / math.sqrt(c), atol=1e-12, rtol=1e-12)
    truth_model = copy.deepcopy(physical)
    for name, val in TRUTH[task].items():
        getattr(truth_model, name).fill_(math.log(val))
    b, a = reference_functions(task, x)
    assert torch.allclose(truth_model.f(0., x), b, atol=1e-12, rtol=1e-12)
    assert torch.allclose(truth_model.g(0., x).square(), a, atol=1e-12, rtol=1e-12)
    return dict(drift_max_abs=ferr, diffusion_max_abs=gerr, truth_formula_match=True)


def reproduce_score():
    task = Task("LV")
    p = Path(ROOT) / "generated/Ours/LV/seed_1.npz"
    pred = np.load(p)["forecast"]
    metrics = mmd2(task.forecast_truth, pred, task.h_train)
    w = emd(task.forecast_truth, pred)
    with open(Path(ROOT) / "results/scores.csv") as f:
        old = [r for r in csv.DictReader(f) if r["method"] == "Ours" and r["task"] == "LV"
               and r["seed"] == "1" and r["kind"] == "forecast"]
    assert len(old) == 1
    diffs = {"mmd2": abs(metrics["snap"] - float(old[0]["mmd2"])),
             "emd": abs(w - float(old[0]["emd"]))}
    assert max(diffs.values()) < 1e-12, diffs
    return dict(rows=1, mmd2=metrics["snap"], emd=w, max_difference=max(diffs.values()),
                prediction_sha256=sha256(p))


def summarize(rows, group_keys, value_keys):
    groups = {}
    for row in rows:
        if row["status"] == "ok":
            groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for key, group in groups.items():
        r = dict(zip(group_keys, key))
        r["n"] = len(group)
        for value in value_keys:
            a = np.array([x[value] for x in group])
            r.update({value + "_mean": float(a.mean()), value + "_sd": float(a.std(ddof=1)),
                      value + "_median": float(np.median(a)), value + "_min": float(a.min()),
                      value + "_max": float(a.max())})
        out.append(r)
    return out


@torch.no_grad()
def run(out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    (out / "tables").mkdir()
    inputs = {}
    def record(p):
        inputs[str(Path(p).resolve())] = sha256(p)
    repo = Path(ROOT).parent
    manifest = dict(start_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
                    git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
                    seeds=SEEDS, truth=TRUTH, scale=SCALE, inputs=inputs)
    for p in [Path(__file__), Path(__file__).with_name("simulate.py"), Path(__file__).with_name("common.py"),
              Path(__file__).parent / "models/classic.py", Path(__file__).parent / "models/missingobs.py",
              repo / "package/snapMMD/booleansde.py", repo / "package/snapMMD/dls.py",
              Path(ROOT) / "fit/snapmmd/classic/classic_sde.py", Path(ROOT) / "fit/snapmmd/missingobs/missing_data.py",
              Path(ROOT) / "results/scores.csv", Path(ROOT) / "generated/Ours/LV/seed_1.npz"]:
        record(p)
    checks = dict(reproduce_first=reproduce_score(), conversions={})
    parameters, functions, failures, fingerprints = [], [], [], []
    for task in ("LV", "ReprParam", "ReprProtein", "ReprSemiparam"):
        data_path = Path(DATA) / (TASKS[task] + ".npz")
        record(data_path)
        assert float(np.load(data_path)["time_scale"]) == SCALE[task]
        initial = MODELS["Ours"][task][0]().double()
        if task in TRUTH:
            checks["conversions"][task] = check_conversion(task, initial)
        x = None
        if task != "ReprProtein":
            val_path = Path(DATA) / (TASKS[task] + "_interp_val.npz")
            record(val_path)
            val = np.load(val_path)
            assert val["Xs"].shape[0:2] == (9, 200)
            x = torch.tensor(val["Xs"].reshape(-1, val["Xs"].shape[-1]), dtype=torch.float64)
            truth_fns = reference_functions(task, x)
        for seed in SEEDS:
            ckpt = Path(ROOT) / "checkpoints/snapmmd" / task / ("model_%d.pt" % seed)
            try:
                record(ckpt)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model = MODELS["Ours"][task][0]().double()
                model.load_state_dict(state, strict=True)
                assert all(torch.isfinite(v).all() for v in state.values())
                fingerprints.append(dict(task=task, seed=seed, file_sha256=sha256(ckpt), tensor_sha256=tensor_hash(state)))
                # Match the released loader in its original evaluation dtype.
                original = MODELS["Ours"][task][0]()
                original.load_state_dict(state, strict=True)
                if task in TRUTH:
                    check_conversion(task, model)
                    for name, truth in TRUTH[task].items():
                        logval = float(getattr(model, name))
                        estimate = math.exp(logval) / SCALE[task] ** power(name)
                        init = math.exp(float(getattr(initial, name))) / SCALE[task] ** power(name)
                        assert getattr(model, name).requires_grad
                        parameters.append(dict(task=task, seed=seed, parameter=name, status="ok", learned="yes",
                            stored_log=logval, solver_value=math.exp(logval), time_power=power(name),
                            estimate=estimate, truth=truth, initial=init, bias=estimate-truth,
                            abs_error=abs(estimate-truth), relative_error=abs(estimate-truth)/truth,
                            log_ratio=math.log(estimate/truth), latent_coordinate_dependent=name in ("beta_p", "k") and task == "ReprProtein"))
                    if task == "ReprProtein":
                        val = math.exp(float(model.beta_p-model.k)) / SCALE[task]
                        ini = math.exp(float(initial.beta_p-initial.k)) / SCALE[task]
                        parameters.append(dict(task=task, seed=seed, parameter="beta_p/k", status="ok", learned="derived",
                            stored_log=float(model.beta_p-model.k), solver_value=val*SCALE[task], time_power=1.,
                            estimate=val, truth=1., initial=ini, bias=val-1, abs_error=abs(val-1),
                            relative_error=abs(val-1), log_ratio=math.log(val), latent_coordinate_dependent=False))
                if x is not None:
                    for name in ("f", "g"):
                        orig = getattr(original, name)(0., x.float()).double()
                        assert torch.allclose(orig, getattr(model, name)(0., x), rtol=1e-4, atol=1e-4)
                    estimates = (model.f(0., x)/SCALE[task], model.g(0., x).square()/SCALE[task])
                    init_fns = (initial.f(0., x)/SCALE[task], initial.g(0., x).square()/SCALE[task])
                    for label, estimate, truth, init in zip(("drift", "covariance"), estimates, truth_fns, init_fns):
                        denominator = float(truth.square().sum())
                        assert denominator > 0 and torch.isfinite(estimate).all()
                        functions.append(dict(task=task, seed=seed, function=label, status="ok", n_states=len(x),
                            rmse=float((estimate-truth).square().mean().sqrt()),
                            nrmse=math.sqrt(float((estimate-truth).square().sum())/denominator),
                            initial_nrmse=math.sqrt(float((init-truth).square().sum())/denominator),
                            truth_sum_squares=denominator, zero_denominator=False,
                            diffusion_misspecified=task == "ReprSemiparam"))
            except Exception as e:
                failures.append(dict(task=task, seed=seed, stage="load_or_evaluate", reason=repr(e)))
        print(task + " complete", flush=True)
    write_csv(out / "tables/parameters_by_fit.csv", parameters)
    write_csv(out / "tables/function_errors_by_fit.csv", functions)
    write_csv(out / "tables/parameter_summary.csv", summarize(parameters, ["task", "parameter"],
              ["estimate", "truth", "initial", "bias", "abs_error", "relative_error", "log_ratio"]))
    write_csv(out / "tables/function_summary.csv", summarize(functions, ["task", "function"], ["rmse", "nrmse", "initial_nrmse"]))
    write_csv(out / "tables/checkpoint_fingerprints.csv", fingerprints)
    checks.update(parameter_rows=len(parameters), function_rows=len(functions), fits=len(fingerprints), failures=len(failures))
    # Rehash to verify evaluation did not change its inputs.
    checks["inputs_unchanged"] = all(sha256(p) == h for p, h in inputs.items())
    assert checks["inputs_unchanged"]
    write_json(out / "reproduction_checks.json", checks)
    write_json(out / "failures.json", failures)
    manifest.update(elapsed_seconds=time.monotonic()-start, end_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "input_manifest.json", manifest)
    print(json.dumps(checks, indent=2), flush=True)
    if failures:
        raise RuntimeError("Evaluation recorded failures; inspect failures.json before summarizing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run(args.out)
