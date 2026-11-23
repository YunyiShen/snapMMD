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

## Learned state-dependent diffusion and learning-rate comparison

The opt-in `--diffusion learned_state` mode retains the paper's
`g(x) = exp(log_sigma) * relu(x)`: **one positive scalar shared across coordinates**,
jointly fitted with the drift. Its initial solver-time amplitude is 0.03,
as in the original structured fits; physical amplitude is solver amplitude
divided by `sqrt(time_scale)`. This differs from both the frozen-true-diffusion
experiment and the published MLP ablation's learned constant diagonal diffusion.
The original `fixed_true` mode remains the default.

Submit the learned-diffusion batch from the repository root using the existing
CUDA environment:

```bash
sbatch reproduction/fit/drift_only/rusty_learned_gpu.sbatch
```

| Array index | Task | Drift | Learning rate | Epochs | Gradient clip |
|---|---|---|---|---|---|
| 0 | ReprParam | Structured, original recipe | 0.05 constant | 800 | None |
| 1–2 | LV, ReprParam | MLP | 0.0025 constant | 1,000 | 1 |
| 3–4 | LV, ReprParam | MLP | 0.001 constant | 1,000 | 1 |
| 5–6 | LV, ReprParam | MLP | 0.01 constant | 1,000 | 1 |
| 7–8 | LV, ReprParam | MLP | 0.01 to 0.0001 cosine | 1,000 | 1 |

All nine jobs are fresh seed-1 fits with explicit Brownian noise, paired between
configurations within each task. The
structured control retains the paper's model, initial guesses, learning rate,
epoch count, absence of clipping, and solver settings. Historical Brownian seeds
were not recorded: comparable fit quality is the control, not identical saved
particles. MLP initialization uses the run seed, whereas the historical MLP
script hard-coded initialization seed 0. Paper checkpoints are never replaced.
All MLP configurations use identical initial weights and diffusion amplitude.

Each job requests one H100, one CPU thread, 8 GB RAM, and two hours; at most two
jobs run together. Maximum allocation is **18 GPU-hours**. Previous measurements
suggest about 12–13 training GPU-hours, but learned-diffusion runtime and queue
availability still need confirmation. A one-seed configuration comparison does
not supply fit-seed uncertainty. No held-out score selects a checkpoint or
training setting; final-epoch predictions and failures are retained for all runs.

An optional short smoke batch must use a separate stage:

```bash
sbatch --time=00:10:00 \
  --export=ALL,SNAPMMD_EPOCHS=20,SNAPMMD_STAGE=learned-state-pilot-gpu \
  reproduction/fit/drift_only/rusty_learned_gpu.sbatch
```

Results go to
`/mnt/ceph/users/$USER/snapmmd-runs/learned-state-full-gpu/<array-id>/`.
Logs are `slurm-drift-learned-<array-id>_<index>.out` in the submission directory.
From any directory, inspect completion and metrics using the actual array ID:

```bash
sacct -j JOB_ID --format=JobID%24,State,Elapsed,ExitCode
snapmmd_run="/mnt/ceph/users/$USER/snapmmd-runs/learned-state-full-gpu/JOB_ID"
tail -n +1 "$snapmmd_run"/*/summary.json
tail -n +1 "$snapmmd_run"/*/metrics.csv
```

The trainer records the learning rate **used** for each update, the scheduler
state, and initial/final sigma in both time units. Cosine scheduling has no
restarts, runs for exactly `--epochs` optimizer steps, and reaches `--min-lr`
after the final update. To resume an interrupted cosine run, retain its original
`--epochs` (the schedule horizon), all other settings, and source checkout;
pass its existing output directory with `--resume`. Do not extend a cosine run
by changing the horizon. Completed outputs and incompatible resumes are rejected.
Old pre-extension checkpoints must be resumed from their original code revision.
`training_summary.json` records completed fitting even if later evaluation fails.
Returned evaluation clouds are saved incrementally to `predictions.partial.npz`
before finiteness and metric checks; successful evaluation replaces this with
`predictions.npz`. A solver exception can prevent the current cloud from being
returned, but earlier clouds, checkpoints, and failure metadata are retained.

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
