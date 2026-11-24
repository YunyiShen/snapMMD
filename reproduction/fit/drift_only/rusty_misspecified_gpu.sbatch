#!/usr/bin/env bash
# Four misspecified-drift fits, one H100 each, two concurrent, <=8 GPU-hours.
# Submit from the repository root with SNAPMMD_EXPECT_COMMIT pinned.
#SBATCH --job-name=drift-misspecified
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
ROOT=${SNAPMMD_REPO:-${SLURM_SUBMIT_DIR:-}}
[[ -n "$ROOT" ]] || { echo "Set SNAPMMD_REPO or submit from the repository root." >&2; exit 1; }
ROOT=$(cd "$ROOT" && pwd -P)
PY=${SNAPMMD_PYTHON:-$HOME/venvs/snapmmd-drift-cuda/bin/python}
RUNS=${SNAPMMD_RUNS:-/mnt/ceph/users/$USER/snapmmd-runs}
STAGE=${SNAPMMD_STAGE:-misspecified-full-gpu}
[[ "$STAGE" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo "SNAPMMD_STAGE must be one lowercase path component (letters, digits, _ or -)." >&2; exit 1; }
ID=${SLURM_ARRAY_TASK_ID:?submit as an array}
[[ "$ID" =~ ^[0-3]$ ]] || { echo "array id must be 0..3" >&2; exit 1; }
TASKS=(LV ReprParam LV ReprParam)
DRIFTS=(affine affine lv_saturated repr_exponential)
TASK=${TASKS[$ID]}
DRIFT=${DRIFTS[$ID]}
EPOCHS=${SNAPMMD_EPOCHS:-1000}
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "SNAPMMD_EPOCHS must be a positive integer." >&2; exit 1; }
if [[ "$EPOCHS" != 1000 && "$STAGE" == misspecified-full-gpu ]]; then
    echo "An epoch override requires a separate SNAPMMD_STAGE." >&2; exit 1
fi
SEED=1
OUT="$RUNS/$STAGE/${SLURM_ARRAY_JOB_ID:?submit as an array}/${TASK}_${DRIFT}_s${SEED}"
CMD=("$PY" "$ROOT/reproduction/fit/drift_only/train.py"
    --task "$TASK" --drift "$DRIFT" --diffusion learned_state --seed "$SEED"
    --epochs "$EPOCHS" --lr 0.01 --schedule constant --gradclip 1.0
    --dt 0.001 --threads 1 --device cuda --output "$OUT")
if [[ ${SNAPMMD_DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
EXPECTED=${SNAPMMD_EXPECT_COMMIT:?set SNAPMMD_EXPECT_COMMIT to the full reviewed commit hash at submission}
ACTUAL=$(git -C "$ROOT" rev-parse HEAD)
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "Checkout changed: expected $EXPECTED, found $ACTUAL" >&2; exit 1; }
git -C "$ROOT" diff --quiet HEAD -- package/snapMMD/dls.py reproduction/code/common.py \
    reproduction/code/models/classic.py reproduction/fit/drift_only/train.py \
    reproduction/fit/drift_only/misspecified.py reproduction/fit/drift_only/rusty_misspecified_gpu.sbatch || {
    echo "Run source has uncommitted changes; use the reviewed commit." >&2; exit 1;
}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" - <<'PY'
import torch, torchsde, numpy, ot
if not torch.cuda.is_available():
    raise RuntimeError("GPU job has no usable CUDA device; refusing a CPU fallback")
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
x = torch.ones(4, 4, device="cuda", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
print("CUDA preflight OK:", torch.cuda.get_device_name(), "runtime", torch.version.cuda)
print("torch", torch.__version__, "torchsde", torchsde.__version__, "numpy", numpy.__version__, "POT", ot.__version__)
PY
echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} task=$TASK drift=$DRIFT output=$OUT source=$ACTUAL"
exec "${CMD[@]}"
