"""Evaluate first/last starts and synthetic horizons from frozen models.

Each invocation writes a new directory. Pilot is seed 1 / one repeat; full uses
the supplied seed list / three repeats. Explicit simulation noise, immutable
inputs, per-row failures, and a wall-clock budget are recorded. No training.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchsde

from common import DATA, ROOT, SEEDS, TASKS, Task, emd, mmd2
from forecast_models import Adapter
from forecast_truth import HORIZONS, TARGET_IDS, TruthSDE, population, streams
from parameter_recovery import sha256, write_csv, write_json

METHODS = ("Ours", "SBIRR-ref", "SB-forward")
FIELDS = ["panel", "task", "method", "seed", "replicate", "start", "horizon", "physical_time",
          "status", "reason", "mmd2", "mmd2_h_train", "emd", "entropy", "seconds", "source", "first_failed_requested_time"]


def valid_prefix(predictions):
    """Keep finite early endpoints if a later forecast diverges."""
    finite = np.isfinite(predictions).all(axis=(1, 2))
    return int(np.flatnonzero(~finite)[0]) if not finite.all() else len(finite)


def score(task, pred, truth):
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("prediction shape/nonfinite check failed")
    m = mmd2(truth, pred, task.h_train)
    return dict(mmd2=m["snap"], mmd2_h_train=m["h_train"], emd=emd(truth, pred))


def historical(out, task_objects, record, jkonet_dir=None):
    """Recompute published forecasts, including last-start SB and persistence."""
    old_path = Path(ROOT) / "results/scores.csv"
    record(old_path)
    with old_path.open() as f:
        old = {(r["method"], r["task"], int(r["seed"])): r for r in csv.DictReader(f) if r["kind"] == "forecast"}
    rows, differences = [], []
    methods = list(METHODS) + ["SBIRR-ref (last snapshot)", "SB-forward (last snapshot)",
                              "Persistence", "JKOnet*", "PRESCIENT", "PI-SDE", "scNODE"]
    for task, obj in task_objects.items():
        for method in methods:
            for seed in SEEDS:
                key = (method, task, seed)
                if key not in old:
                    continue
                path = Path(ROOT) / ("generated" if method == "Ours" else "outputs") / method / task / ("seed_%d.npz" % seed)
                if method == "Persistence":
                    pred = obj.train[-1]
                else:
                    record(path)
                    pred = np.load(path)["forecast"]
                values = score(obj, pred, obj.forecast_truth)
                diff = max(abs(values[k] - float(old[key][k])) for k in ("mmd2", "mmd2_h_train", "emd"))
                differences.append(diff)
                rows.append(dict(panel="historical", task=task, method=method, seed=seed, replicate="historical",
                    start="na" if method == "Persistence" else ("last" if "last snapshot" in method else "first"),
                    horizon=1, physical_time="", status="ok", reason="", entropy="", seconds="", source=str(path), **values))
        if jkonet_dir:
            source_task = "ReprParam" if task == "ReprSemiparam" else task
            for seed in SEEDS:
                path = Path(jkonet_dir) / ("%s_forecast_%d.npz" % (source_task, seed))
                record(path)
                a = np.load(path)
                released = np.load(Path(ROOT) / "outputs/JKOnet*" / task / ("seed_%d.npz" % seed))["forecast"]
                assert np.array_equal(a["forecast"][-1], released), "JKOnet* historical configuration mismatch"
                values = score(obj, a["forecast_onestep"], obj.forecast_truth)
                rows.append(dict(panel="historical", task=task, method="JKOnet*", seed=seed, replicate="historical",
                    start="last", horizon=1, physical_time="", status="ok", reason="one saved evaluation draw",
                    entropy="", seconds="", source=str(path), **values))
        print("Historical scores verified: " + task, flush=True)
    assert differences and max(differences) < 1e-10, (len(differences), max(differences))
    write_csv(out / "tables/historical_scores.csv", rows)
    return dict(rows_compared=len(differences), max_abs_difference=max(differences), saved_restart_rows=len(rows)-len(differences))


def entropy(task, method, seed, rep):
    # Stable, explicit substreams. Same entropy for both starts within a fitted model.
    return int(np.random.SeedSequence([91001, list(TASKS).index(task), METHODS.index(method) if method in METHODS else 9,
                                      seed, rep]).generate_state(1)[0])


def run(args):
    start = time.monotonic()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "tables").mkdir()
    (out / "outputs").mkdir()
    inputs, failures, timing, checks = {}, [], [], {}
    def record(p):
        p = Path(p).resolve()
        inputs[str(p)] = sha256(p)
    def remaining():
        return args.budget_seconds - (time.monotonic() - start)
    repo = Path(ROOT).parent
    manifest = dict(start_utc=datetime.now(timezone.utc).isoformat(), args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        code_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        python=platform.python_version(), torch=torch.__version__, torchsde=torchsde.__version__, numpy=np.__version__,
        threads=torch.get_num_threads(), seeds=args.seeds, input_hashes=inputs,
        target_replicates=list(TARGET_IDS), model_dt=.001, truth_physical_dt=.001, model_clocks={},
        brownian=dict(halfway_tree=True, tol=1e-6, future_coupling="same entropy/domain within each model"))
    source_paths = [Path(__file__), Path(__file__).with_name("forecast_models.py"), Path(__file__).with_name("forecast_truth.py"),
                    Path(__file__).with_name("parameter_recovery.py"), Path(__file__).with_name("common.py"),
                    Path(__file__).with_name("simulate.py"), repo / "package/snapMMD/booleansde.py"]
    source_paths += list((Path(__file__).parent / "models").rglob("*.py"))
    source_paths += list((Path(ROOT) / "fit/baselines/schrodinger_bridge").rglob("*.py"))
    if args.sb_package:
        source_paths += list(Path(args.sb_package).rglob("*.py"))
    if args.sb_stubs:
        source_paths += list(Path(args.sb_stubs).rglob("*.py"))
    for p in source_paths:
        record(p)
    task_objects, data = {}, {}
    for task in TASKS:
        p = Path(DATA) / (TASKS[task] + ".npz")
        record(p)
        record(Path(DATA) / (TASKS[task] + "_interp_val.npz"))
        data[task] = dict(np.load(p))
        task_objects[task] = Task(task)
    checks["historical"] = historical(out, task_objects, record, args.jkonet_dir)
    write_json(out / "reproduction_checks.json", checks)
    write_json(out / "input_manifest.json", manifest)
    repeats = 1 if args.pilot else 3
    seeds = [1] if args.pilot else args.seeds
    targets, target_meta, target_hashes = {}, {}, {}
    target_tasks = ("LV",) if args.pilot else ("LV", "ReprParam")
    target_horizons = (1,) if args.pilot else HORIZONS
    for task in target_tasks:
        for rep in range(repeats):
            for h in target_horizons:
                if remaining() < 30:
                    raise TimeoutError("Budget exhausted before target generation completed")
                t = time.monotonic()
                a, meta = population(task, h, TARGET_IDS[rep])
                targets[task, rep, h] = a
                target_hashes[task, rep, h] = hashlib.sha256(a.tobytes()).hexdigest()
                target_meta["%s_r%d_h%d" % (task,rep,h)] = meta
                np.savez_compressed(out / "outputs" / ("truth_%s_r%d_h%d.npz" % (task, rep, h)), truth=a)
                timing.append(dict(stage="truth", task=task, method="generator", seconds=time.monotonic()-t))
    if args.pilot:
        half, _ = population("LV", 1, TARGET_IDS[0], dt=.0005)
        full = targets["LV", 0, 1]
        checks["truth_half_step"] = dict(coupled_particle_rmse=float(np.sqrt(np.mean((full-half)**2))),
            emd=emd(full, half), mmd2=mmd2(full, half)["snap"])
    write_json(out / "target_manifest.json", target_meta)
    score_path = out / "tables/forecast_scores.csv"
    with score_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        def emit(base, pred=None, truth=None):
            row = {k:"" for k in FIELDS}
            row.update(base)
            try:
                if pred is not None:
                    row.update(score(task_objects[row["task"]], pred, truth))
                    row["status"] = "ok"
            except Exception as e:
                row.update(status="failed", reason=repr(e))
            if row["status"] != "ok":
                failures.append({k:row[k] for k in ("panel","task","method","seed","replicate","start","horizon","physical_time","status","reason","first_failed_requested_time")})
            writer.writerow(row)
            f.flush()
        # Persistence has no fitted-seed replication; keep this separate in summaries.
        for task, obj in task_objects.items():
            phy = 20. if task == "PBMC" else 1.
            emit(dict(panel="A", task=task, method="Persistence", seed="none", replicate="fixed", start="na", horizon=1,
                      physical_time=float(data[task]["dts"][-1])*phy), obj.train[-1], obj.forecast_truth)
        for (task, rep, h), truth in targets.items():
            emit(dict(panel="B", task=task, method="Persistence", seed="none", replicate=rep, start="na", horizon=h,
                      physical_time=9+h), task_objects[task].train[-1], truth)
        for task, obj in task_objects.items():
            horizons = HORIZONS if task in ("LV", "ReprParam") else (1,)
            physical_factor = 20. if task == "PBMC" else 1.
            endpoints = [9.+h for h in horizons] if task in ("LV", "ReprParam") else [float(data[task]["dts"][-1])*physical_factor]
            for method in METHODS:
                for seed in seeds:
                    adapter, load_error = None, None
                    try:
                        adapter = Adapter(method, task, seed, data[task], args.sb_package, args.sb_stubs)
                        record(adapter.checkpoint)
                        manifest["model_clocks"][task + "/" + method] = dict(
                            physical_to_solver_divisor=adapter.clock_divisor,
                            stored_to_physical_factor=adapter.physical_factor, dtype=str(adapter.dtype),
                            latent_dimensions=adapter.latent)
                    except Exception as e:
                        load_error = repr(e)
                    for rep in range(repeats):
                        ent = entropy(task, method, seed, rep)
                        for protocol in ("first", "last"):
                            t = time.monotonic()
                            predictions, ts, prefix, first_failed = None, None, 0, ""
                            status, reason = "ok", ""
                            if load_error:
                                status, reason = "failed", load_error
                            elif protocol == "last" and adapter.latent:
                                status, reason = "unavailable", "unobserved latent state requires separate conditional inference"
                            elif remaining() < 30:
                                status, reason = "budget_not_run", "reserved 30 seconds for finalization"
                            else:
                                try:
                                    predictions, ts = adapter.rollout(data[task], protocol, endpoints, ent)
                                    prefix = valid_prefix(predictions)
                                    if prefix < len(endpoints):
                                        first_failed = endpoints[prefix]
                                    name = "%s_%s_s%d_r%d_%s" % (task, method, seed, rep, protocol)
                                    np.savez_compressed(out / "outputs" / (name + ".npz"), predictions=predictions,
                                                        solver_times=ts, physical_times=np.array(endpoints), entropy=ent)
                                except Exception as e:
                                    status, reason = "failed", repr(e)
                            elapsed = time.monotonic()-t
                            base = dict(task=task, method=method, seed=seed, replicate=rep, start=protocol,
                                        entropy=ent, status=status, reason=reason, seconds=elapsed, source="fresh_explicit_noise",
                                        first_failed_requested_time=first_failed)
                            def endpoint_row(j, panel, h, truth):
                                row = dict(base, panel=panel, horizon=h, physical_time=endpoints[j])
                                valid = status == "ok" and j < prefix
                                if status == "ok" and not valid:
                                    row.update(status="failed", reason="nonfinite rollout; first failed requested time " + str(first_failed))
                                emit(row, predictions[j] if valid else None, truth)
                            endpoint_row(0, "A", 1, obj.forecast_truth)
                            for j,h in enumerate(horizons):
                                if (task,rep,h) in targets:
                                    endpoint_row(j, "B", h, targets[task,rep,h])
                            log_status = "partial_nonfinite" if status == "ok" and prefix < len(endpoints) else status
                            timing.append(dict(stage="rollout", task=task, method=method, seed=seed, replicate=rep,
                                               start=protocol, status=log_status, seconds=elapsed, valid_endpoints=prefix,
                                               first_failed_requested_time=first_failed))
                            print("%s %s s%d r%d %s: %s %.2fs; budget %.0fs" %
                                  (task,method,seed,rep,protocol,log_status,elapsed,remaining()), flush=True)
                            if args.pilot and method == "Ours" and task in ("LV","PBMC") and status == "ok" and prefix:
                                half, _ = adapter.rollout(data[task], protocol, endpoints, ent, dt=.0005)
                                checks.setdefault("model_half_step", []).append(dict(task=task, start=protocol,
                                    coupled_particle_rmse=float(np.sqrt(np.mean((predictions[0]-half[0])**2))),
                                    full_score=score(obj,predictions[0],obj.forecast_truth),
                                    half_score=score(obj,half[0],obj.forecast_truth)))
        # Oracle: same empirical start populations, true physical dynamics, separate RNG.
        for task in target_tasks:
            for rep in range(repeats):
                for protocol in ("first", "last"):
                    endpoints = [9.+h for h in target_horizons]
                    pred, reason = None, ""
                    ent = int(np.random.SeedSequence([92001, list(TASKS).index(task), rep]).generate_state(1)[0])
                    try:
                        if remaining() < 30:
                            raise TimeoutError("budget exhausted")
                        x = torch.tensor(data[task]["Xs"][0 if protocol == "first" else -2], dtype=torch.float64)
                        start_time = 0. if protocol == "first" else 9.
                        bm = torchsde.BrownianInterval(t0=0.,t1=endpoints[-1],size=x.shape,dtype=x.dtype,
                                                       entropy=ent,halfway_tree=True,tol=1e-6)
                        with torch.no_grad():
                            ys = torchsde.sdeint(TruthSDE(task), x, torch.tensor([start_time]+endpoints,dtype=x.dtype),
                                                bm=bm,method="euler",dt=.001,adaptive=False)
                        assert torch.equal(ys[0],x)
                        pred = ys[1:].numpy()
                        np.savez_compressed(out/"outputs"/("oracle_%s_r%d_%s.npz"%(task,rep,protocol)),predictions=pred,physical_times=endpoints)
                    except Exception as e:
                        reason = repr(e)
                    for j,h in enumerate(target_horizons):
                        emit(dict(panel="B",task=task,method="Oracle",seed="none",replicate=rep,start=protocol,
                            horizon=h,physical_time=9+h,entropy=ent,status="failed" if reason else "ok",reason=reason),
                            None if reason else pred[j], targets[task,rep,h])
    checks["inputs_unchanged"] = all(sha256(p)==h for p,h in inputs.items())
    checks["generated_targets_unchanged"] = all(hashlib.sha256(a.tobytes()).hexdigest()==target_hashes[k] for k,a in targets.items())
    assert checks["generated_targets_unchanged"]
    assert checks["inputs_unchanged"]
    write_json(out / "reproduction_checks.json", checks)
    write_json(out / "failures.json", failures)
    write_json(out / "timing.json", timing)
    manifest.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                    status="completed_with_exclusions" if failures else "completed")
    write_json(out / "input_manifest.json", manifest)
    print("Finished; elapsed %.1fs; %d exclusions/failures" % (manifest["elapsed_seconds"],len(failures)),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--seeds",type=int,nargs="+",default=SEEDS)
    ap.add_argument("--budget-seconds",type=float,default=7200)
    ap.add_argument("--sb-package",type=Path)
    ap.add_argument("--sb-stubs",type=Path)
    ap.add_argument("--jkonet-dir",type=Path)
    args=ap.parse_args()
    torch.set_num_threads(1)
    run(args)
