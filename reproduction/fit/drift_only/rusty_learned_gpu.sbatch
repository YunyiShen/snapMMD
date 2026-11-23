#!/usr/bin/env bash
# Eight MLP fits plus the original structured repressilator recipe.
# One H100 per task, at most two tasks together, <=18 GPU-hours total.
# Submit from the repository root; writes only to a new job-ID directory.
#SBATCH --job-name=drift-learned
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
ROOT=${SNAPMMD_REPO:-${SLURM_SUBMIT_DIR:-}}
[[ -n "$ROOT" ]] || { echo "Set SNAPMMD_REPO or submit from the repository root." >&2; exit 1; }
ROOT=$(cd "$ROOT" && pwd -P)
PY=${SNAPMMD_PYTHON:-$HOME/venvs/snapmmd-drift-cuda/bin/python}
RUNS=${SNAPMMD_RUNS:-/mnt/ceph/users/$USER/snapmmd-runs}
STAGE=${SNAPMMD_STAGE:-learned-state-full-gpu}
ID=${SLURM_ARRAY_TASK_ID:?submit as an array}
[[ "$ID" =~ ^[0-8]$ ]] || { echo "array id must be 0..8" >&2; exit 1; }
TASKS=(ReprParam LV ReprParam LV ReprParam LV ReprParam LV ReprParam)
VARIANTS=(paper_control baseline baseline lower lower higher higher cosine cosine)
LRS=(0.05 0.0025 0.0025 0.001 0.001 0.01 0.01 0.01 0.01)
TASK=${TASKS[$ID]}
VARIANT=${VARIANTS[$ID]}
LR=${LRS[$ID]}
DRIFT=mlp
CLIP=1.0
SCHEDULE=constant
EPOCHS=1000
if [[ "$ID" == 0 ]]; then DRIFT=structured; CLIP=none; EPOCHS=800; fi
if (( ID >= 7 )); then SCHEDULE=cosine; fi
# Optional smoke override. Use a separate SNAPMMD_STAGE for a short pilot.
EPOCHS=${SNAPMMD_EPOCHS:-$EPOCHS}
SEED=1
OUT="$RUNS/$STAGE/${SLURM_ARRAY_JOB_ID:?submit as an array}/${TASK}_${DRIFT}_${VARIANT}_s${SEED}"
CMD=("$PY" "$ROOT/reproduction/fit/drift_only/train.py"
    --task "$TASK" --drift "$DRIFT" --diffusion learned_state --seed "$SEED"
    --epochs "$EPOCHS" --lr "$LR" --schedule "$SCHEDULE" --min-lr 0.0001
    --gradclip "$CLIP" --dt 0.001 --threads 1 --device cuda --output "$OUT")
if [[ ${SNAPMMD_DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" - <<'PY'
import torch, torchsde, numpy, ot
if not torch.cuda.is_available():
    raise RuntimeError("GPU job has no usable CUDA device; refusing a CPU fallback")
torch.use_deterministic_algorithms(True)
x = torch.ones(4, 4, device="cuda", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
print("CUDA preflight OK:", torch.cuda.get_device_name(), "runtime", torch.version.cuda)
print("torch", torch.__version__, "torchsde", torchsde.__version__, "numpy", numpy.__version__, "POT", ot.__version__)
PY
echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} task=$TASK variant=$VARIANT output=$OUT"
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short
exec "${CMD[@]}"
