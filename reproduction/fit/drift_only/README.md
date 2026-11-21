# Drift-family comparison with fixed diffusion

Two models fit the same released LV or repressilator snapshots with the same
SnapMMD objective and **known, frozen state-dependent diffusion**. The structured
drift learns its original four mechanistic coefficients; the other drift is the
existing one-hidden-layer, width-64 ReLU MLP with a linear output. Both are
autonomous. This compares drift families within SnapMMD, not fitting objectives.

The original `../fully_neural/MLP.py` also changes diffusion to a learned constant
diagonal. These fits instead share physical diffusion `g(x)=0.02*relu(x)`,
diagonally. With solver time `s=t/c`, the common wrapper multiplies it by `sqrt(c)`.
The source generators use `c=9` for LV and `c=10` for the repressilator.

Training uses the first ten snapshots (physical times 0–9), 1,000 simulated
particles from five repetitions of the first observed cloud, float32, Euler
`dt=0.001` in solver time, adjoint gradients, the released MMD kernel/objective,
Adam with lr 0.0025, and gradient clipping at 1.0. Both arms use the same epoch
budget. Seed the MLP initialization and explicitly seed the shared Brownian
path before training. A given fit reuses its path across epochs.

The historical structured models used lr 0.05 and 300/800 epochs. This new
comparison is a matched-budget experiment, not a guarantee of equal convergence.
Check the loss curves before interpreting a gap. A one-seed pilot cannot support
a claim of superiority or equivalence.

## Local verification

From the repository root, with the reproduction environment:

```bash
python reproduction/fit/drift_only/test_invariants.py -v
python reproduction/fit/drift_only/train.py \
  --task LV --drift mlp --seed 1 --epochs 1 --output /tmp/drift-smoke-LV-mlp
```

The output directory must be new. The tests check architecture, frozen diffusion
and units, exclusion of the forecast target, exact agreement with the released
optimizer step under common noise, and checkpoint/Brownian reproducibility.

## Rusty setup and submission

Clone the working branch to a new directory. A sparse checkout avoids fetching
the old checkpoints and trajectory arrays, which this experiment does not need:

```bash
mkdir -p "$HOME/code"
git clone --filter=blob:none --sparse --single-branch --branch aistats-revision \
  https://github.com/YunyiShen/snapMMD.git "$HOME/code/snapmmd-drift"
cd "$HOME/code/snapmmd-drift"
git sparse-checkout set package data/classic reproduction/code \
  reproduction/fit/fully_neural reproduction/fit/drift_only
```

Subsequent runs use `git pull --ff-only` before submission. Record the commit
and avoid pulling again while a queued job might still read the checkout.
For concurrent revisions, use separate clean git worktrees pinned to commits.

Create an isolated CPU environment once:

```bash
bash reproduction/fit/drift_only/setup_rusty.sh
```

This uses Rusty's `python/3.11.11` module and installs torch 2.2.1 (CPU),
torchsde 0.2.6, NumPy 1.23.5, SciPy 1.10.1, POT 0.9.7.post1, and tqdm.
It refuses to overwrite an existing environment. Versions are saved in
`~/venvs/snapmmd-drift/requirements.lock.txt`. The module and installation
are adapted from recent Rusty scripts; they still need verification there.
The installer needs network access for package downloads.

Run numerical checks on a compute node:

```bash
srun -p genx -n 1 -c 1 --mem-per-cpu=4G -t 00:05:00 \
  "$HOME/venvs/snapmmd-drift/bin/python" reproduction/fit/drift_only/test_invariants.py -v
```

Submit the 20-epoch pilot:

```bash
sbatch reproduction/fit/drift_only/rusty.sbatch
```

This submits four array tasks: LV structured, LV MLP, ReprParam structured,
ReprParam MLP, all seed 1; at most two run concurrently. Each requests one
core on shared partition `genx`, 4 GB per core and 30 minutes: at most
**2 allocated CPU-hours** total, no GPU. BLAS and torch use one thread.
Do not use whole-node partition `gen` for this workload.

Check the logs, failures, memory usage and epoch times before the full run.
One-epoch laptop checks took about 2.1–2.4 seconds per training epoch;
these are not Rusty timings. At that rate the four 1,000-epoch fits would use
roughly 2.5 CPU-hours in total. If the cluster pilot fits the two-hour per-fit
limit with margin, submit:

```bash
sbatch --time=02:00:00 \
  --export=ALL,SNAPMMD_EPOCHS=1000,SNAPMMD_STAGE=full \
  reproduction/fit/drift_only/rusty.sbatch
```

The full seed-1 array is capped at **8 allocated CPU-hours**, two concurrent
cores, no GPU. A ten-seed expansion would use array `0-39%2`; it is deliberately
not the default and needs a separate budget decision.

Overrides: `SNAPMMD_REPO` (code root), `SNAPMMD_PYTHON` (Python executable),
`SNAPMMD_RUNS` (output root), `SNAPMMD_STAGE`, `SNAPMMD_EPOCHS`, `SNAPMMD_DEVICE`.
Defaults write results under `/mnt/ceph/users/$USER/snapmmd-runs/<stage>/<array-id>/`.
The launcher does not send email or interact with other projects.

## GPU pilot

The MLPs have 322 (LV) and 451 (repressilator) trainable parameters, but each epoch
also simulates 1,000 particles and computes dense MMD kernels. A GPU may help;
the many sequential solver steps also impose overhead. CPU feasibility alone
does not establish which backend is faster. Compare the same 20-epoch pilot
on Rusty CPU and GPU before choosing the full-run backend.

Create a **separate** CUDA environment once, then submit from the repository root:

```bash
bash reproduction/fit/drift_only/setup_rusty.sh cuda
sbatch reproduction/fit/drift_only/rusty_gpu.sbatch
```

The installer uses `~/venvs/snapmmd-drift-cuda` and the official torch 2.2.1
[CUDA 12.1 wheel](https://pytorch.org/get-started/previous-versions/#v221).
It leaves the CPU environment unchanged. A CUDA-enabled PyTorch installation
is necessary; allocating a GPU while using the CPU wheel is insufficient.

The GPU launcher requests **one H100 and one CPU core**, 8 GB host RAM, 10 minutes
per fit, four tasks with concurrency one. Total cap: **40 GPU-minutes and
40 CPU-minutes**. H100 matches existing Rusty node templates and the pinned
CUDA build; these small MLPs do not require that particular GPU model.
Outputs use stage `pilot-gpu`. CUDA availability and a small forward/backward
operation are checked before training; a missing/misconfigured GPU aborts the
job, with no CPU fallback. GPU model, driver and CPU information go in the log.

After both pilots, compare `median_epoch_seconds_after_5_warmup` in each fit's
`summary.json`, overall training time, failures and memory. That median uses
epochs 6–20 for a fresh 20-epoch run. CUDA timing explicitly synchronizes work;
see [PyTorch's timing guidance](https://docs.pytorch.org/docs/2.2/notes/cuda.html#asynchronous-execution).
GPU peak allocated tensor memory is saved as `cuda_peak_allocated_bytes`.
Evaluation/serialization overhead and queue time are separate from epoch time.
The full GPU resource request should be selected from these measurements;
the launcher defaults only to the short pilot.

Both backends retain float32 and deterministic algorithms. GPU runs explicitly
disable TF32 and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, as required for deterministic
CUDA matrix operations in [PyTorch 2.2](https://docs.pytorch.org/docs/2.2/generated/torch.use_deterministic_algorithms.html).
CPU and CUDA runs need not be bit-identical even with the same seed. Use the
same backend for both drift arms when making the scientific comparison.

The CUDA runtime and cluster execution still need validation on Rusty. Local
CPU regression checks do not constitute a GPU benchmark.

## Outputs and restart

Each fit saves:

- `config.json`: settings, data/source hashes, package versions and Brownian seed.
- `invocations.jsonl`: command, Git commit/status, host, job ID and requested epochs.
- `history.csv`: pre-update loss, unclipped gradient norm and seconds per epoch.
- `checkpoint.pt`: model and optimizer, every 25 epochs and on completion.
- `predictions.npz`, `metrics.csv`: held-out forecasts from both the first and
  last training snapshots, interpolation at the nine exact target times, and
  last-snapshot persistence. MMD² and W2 reuse the released evaluation functions.
- `summary.json`: completion marker, final post-update training loss, timing
  median after five warmup epochs, and GPU peak allocated memory when applicable.
- `failure_*.json`: exceptions, retained even if a later restart succeeds.

Evaluation uses separate noise streams from training, paired between the arms.
Forecast/interpolation targets are read only after fitting for scoring.

An interrupted fit can be resumed with the same command and `--resume`; pass
the original output directory and original settings/environment. Resumption
rejects changed source/data hashes or configuration. It preserves an audit copy
of the history before dropping rows beyond the last durable checkpoint.
Completed outputs cannot be resumed or overwritten. For a new budget/setting,
use a new directory. A Slurm resubmission gets a new job ID and therefore starts
a fresh fit by default; use the CLI explicitly to resume the old directory.
Updating the trainer changes its source hash: finish/resume existing runs from
their original checkout; use a new output directory for this GPU-capable revision.
